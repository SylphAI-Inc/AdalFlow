from typing import List, Optional, Union, Dict, Any
from uuid import UUID
import uuid
import os
import dotenv
import numpy as np
from abc import ABC, abstractmethod

import backoff
from openai import (
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
    UnprocessableEntityError,
)
from copy import deepcopy

import faiss
import tiktoken


from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline


dotenv.load_dotenv(dotenv_path=".env", override=True)


# TODO: (1) improve logging
##############################################
# Key data structures for RAG
# TODO: visualize the data structures
##############################################
class Document:
    meta_data: dict  # can save data for filtering at retrieval time too
    text: str
    id: Optional[Union[str, UUID]] = (
        None  # if the file name is unique, its better to use it as id instead of UUID
    )
    estimated_num_tokens: Optional[int] = (
        None  # useful for cost and chunking estimation
    )

    def __init__(
        self,
        meta_data: dict,
        text: str,
        id: Optional[Union[str, UUID]] = None,
        estimated_num_tokens: Optional[int] = None,
    ):
        self.meta_data = meta_data
        self.text = text
        self.id = id
        self.estimated_num_tokens = estimated_num_tokens

    @staticmethod
    def from_dict(doc: Dict):
        assert "meta_data" in doc, "meta_data is required"
        assert "text" in doc, "text is required"
        if "estimated_num_tokens" not in doc:
            tokenizer = Tokenizer()
            doc["estimated_num_tokens"] = tokenizer.count_tokens(doc["text"])
        if "id" not in doc:
            doc["id"] = uuid.uuid4()

        return Document(**doc)

    def __repr__(self) -> str:
        return f"Document(id={self.id}, meta_data={self.meta_data}, text={self.text[0:50]}, estimated_num_tokens={self.estimated_num_tokens})"

    def __str__(self):
        return self.__repr__()


class Chunk:
    vector: List[float]
    text: str
    order: Optional[int] = None  # order of the chunk in the document

    doc_id: Optional[Union[str, UUID]] = (
        None  # id of the Document where the chunk is from
    )
    id: Optional[Union[str, UUID]] = None
    estimated_num_tokens: Optional[int] = None
    score: Optional[float] = None  # used in retrieved output
    meta_data: Optional[Dict] = (
        None  # only when the above fields are not enough or be used for metadata filtering
    )

    def __init__(
        self,
        vector: List[float],
        text: str,
        order: Optional[int] = None,
        doc_id: Optional[Union[str, UUID]] = None,
        id: Optional[Union[str, UUID]] = None,
        estimated_num_tokens: Optional[int] = None,
        meta_data: Optional[Dict] = None,
    ):
        self.vector = vector if vector else []
        self.text = text
        self.order = order
        self.doc_id = doc_id
        self.id = id if id else uuid.uuid4()
        self.meta_data = meta_data

        self.estimated_num_tokens = estimated_num_tokens if estimated_num_tokens else 0
        # estimate the number of tokens
        if not self.estimated_num_tokens:
            tokenizer = Tokenizer()
            self.estimated_num_tokens = tokenizer.count_tokens(self.text)

    def __repr__(self) -> str:
        return f"Chunk(id={self.id}, doc_id={self.doc_id}, order={self.order}, text={self.text}, vector={self.vector[0:5]}, estimated_num_tokens={self.estimated_num_tokens}, score={self.score})"

    def __str__(self):
        return self.__repr__()


##############################################
# Helper modules for RAG
##############################################
class Tokenizer:
    def __init__(self, name: str = "cl100k_base"):
        self.name = name
        self.tokenizer = tiktoken.get_encoding(name)

    def encode(self, text: str) -> List[str]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[str]) -> str:
        return self.tokenizer.decode(tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))


##############################################
# Key functional modules for RAG
##############################################
class Embedder(ABC):
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    @abstractmethod
    def __call__(self, queries: Union[str, List[str]]) -> Any:
        pass


class EmbedderOutput:
    embeddings: List[List[float]]
    usage: Dict[str, Any]  # api or model usage

    def __init__(self, embeddings: List[List[float]], usage: Dict[str, Any]):
        self.embeddings = embeddings
        self.usage = usage

    def __repr__(self) -> str:
        return f"EmbedderOutput(embeddings={self.embeddings[0:5]}, usage={self.usage})"

    def __str__(self):
        return self.__repr__()


class OpenAIEmbedder(Embedder):
    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__(provider, model)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Environment variable OPENAI_API_KEY must be set")
        self.client = OpenAI()
        self.kwargs = kwargs

    @staticmethod
    def _process_text(text: str) -> str:
        """
        This is specific to OpenAI API, as removing new lines could have better performance
        """
        text = text.replace("\n", " ")
        return text

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
        ),
        max_time=5,
    )
    def __call__(
        self,
        queries: Union[str, List[str]],
    ) -> EmbedderOutput:
        """
        Automatically handles retries for the above exceptions
        TODO: support async calls
        """
        formulated_queries = []
        if isinstance(queries, str):
            formulated_queries.append(self._process_text(queries))
        else:
            for query in queries:
                formulated_queries.append(self._process_text(query))

        num_queries = len(formulated_queries)

        print(f"kewargs: {self.kwargs}, self.model: {self.model}")

        response = self.client.embeddings.create(
            input=formulated_queries, model=self.model, **self.kwargs
        )
        usage = response.usage
        embeddings = [data.embedding for data in response.data]
        assert (
            len(embeddings) == num_queries
        ), f"Number of embeddings {len(embeddings)} is not equal to the number of queries {num_queries}"
        return EmbedderOutput(embeddings=embeddings, usage=usage)


class Retriever(ABC):
    name = "Retriever"
    input_variable = "query"
    desc = "takes a search query and returns one or more potentially relevant passages from a corpus"

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    # def reset(self):
    #     pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class RetrieverOutput:
    """
    Retrieved result per query
    """

    chunks: List[Chunk]
    query: Optional[str] = None

    def __init__(self, chunks: List[Chunk], query: Optional[str] = None):
        self.chunks = chunks
        self.query = query

    def __repr__(self) -> str:
        return f"RetrieverOutput(chunks={self.chunks[0:5]}, query={self.query})"

    def __str__(self):
        return self.__repr__()


class Generator(ABC):
    name = "Generator"
    input_variable = "query"
    desc = "Takes in query + context and generates the answer"

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    # def reset(self):
    #     pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class FAISSRetriever(Retriever):
    """
    https://github.com/facebookresearch/faiss
    The retriever uses in-memory Faiss index to retrieve the top k chunks
    d: dimension of the vectors
    xb: number of vectors to put in the index
    xq: number of queries
    The data type dtype must be float32.
    Note: When the num of chunks are less than top_k, the last columns will be -1

    Other index options:
    - faiss.IndexFlatL2: L2 or Euclidean distance, [-inf, inf]
    - faiss.IndexFlatIP: inner product of normalized vectors will be cosine similarity, [-1, 1]

    We choose cosine similarity and convert it to range [0, 1] by adding 1 and dividing by 2 to simulate probability
    """

    name = "FAISSRetriever"

    def __init__(
        self,
        top_k: int = 3,
        d: int = 768,
        chunks: Optional[List[Chunk]] = None,
        vectorizer: Optional[Embedder] = None,
    ):
        self.d = d
        self.index = faiss.IndexFlatIP(
            d
        )  # inner product of normalized vectors will be cosine similarity, [-1, 1]

        self.vectorizer = vectorizer  # used to vectorize the queries
        if chunks:
            self.set_chunks(chunks)
        super().__init__(top_k)

    # def reset(self):
    #     self.index.reset()
    #     self.chunks = []

    def set_chunks(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.total_chunks = len(chunks)
        embeddings = [chunk.vector for chunk in chunks]
        xb = np.array(embeddings, dtype=np.float32)
        self.index.add(xb)

    def _convert_cosine_similarity_to_probability(self, D: np.ndarray) -> np.ndarray:
        D = (D + 1) / 2
        D = np.round(D, 3)
        return D

    def _to_retriever_output(
        self, I: np.ndarray, D: np.ndarray
    ) -> List[RetrieverOutput]:
        output: List[RetrieverOutput] = []
        # Step 1: Filter out the -1, -1 areas along with its scores
        if -1 in I:
            valid_columns = ~np.any(I == -1, axis=0)

            # Filter out rows where the last two columns
            D = D[:, valid_columns]
            I = I[:, valid_columns]
        # Step 2: processing rows (one query at a time)
        for row in zip(I, D):
            indexes, distances = row
            chunks: List[Chunk] = []
            for index, distance in zip(indexes, distances):
                chunk = deepcopy(self.chunks[index])
                chunk.score = distance
                chunks.append(chunk)

            output.append(RetrieverOutput(chunks=chunks))

        return output

    def __call__(
        self, query_or_queries: Union[str, List[str]], top_k: Optional[int] = None
    ) -> List[RetrieverOutput]:
        # if you pass a single query, you should access the first element of the list
        if self.index.ntotal == 0:
            raise ValueError(
                "Index is empty. Please set the chunks to build the index from"
            )
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [q for q in queries if q]  # Filter empty queries
        queries_embeddings = self.vectorizer(queries).embeddings
        xq = np.array(queries_embeddings, dtype=np.float32)
        D, I = self.index.search(xq, top_k if top_k else self.top_k)
        D = self._convert_cosine_similarity_to_probability(D)
        retrieved_output = self._to_retriever_output(I, D)
        for i, output in enumerate(retrieved_output):
            output.query = queries[i]
        return retrieved_output


##############################################
# Main RAG class
# Configs and combines functional modules
# One settings per RAG instance instead of global settings
##############################################
class RAG:
    """
    TODO: design a base class later
    inputs: A list of documents [Can potentially use pandas dataframe for more complex data]
    Focus on retrieving for now
    """

    def __init__(self, settings: dict = {}):
        super().__init__()
        self.settings = settings if settings else self.default_settings()
        self.vectorizer = None
        # initialize vectorizer
        if "vectorizer" in self.settings:
            vectorizer_settings = self.settings["vectorizer"]
            if vectorizer_settings["provider"] == "openai":
                kwargs = {}
                # encoding_format and dimensions are optional
                if "encoding_format" in vectorizer_settings:
                    kwargs["encoding_format"] = vectorizer_settings["encoding_format"]
                if (
                    "embedding_size" in vectorizer_settings
                    and "text-embedding-3" in vectorizer_settings["model"]
                ):
                    # only for text-embedding-3-small and later
                    kwargs["dimensions"] = vectorizer_settings["embedding_size"]
                self.vectorizer = OpenAIEmbedder(
                    provider=vectorizer_settings["provider"],
                    model=vectorizer_settings["model"],
                    **kwargs,
                )
        # initialize retriever, which depends on the vectorizer too
        self.retriever = None
        if "retriever_type" in self.settings:
            if self.settings["retriever_type"] == "dense_retriever":
                dense_retriever_settings = self.settings["retriever"]
                if dense_retriever_settings["provider"] == "faiss":
                    self.retriever = FAISSRetriever(
                        top_k=dense_retriever_settings["top_k"],
                        d=vectorizer_settings["embedding_size"],
                        vectorizer=self.vectorizer,
                    )
        self.tracking = {"vectorizer": {"num_calls": 0, "num_tokens": 0}}

    def set_settings(self, settings: dict):
        self.settings = settings

    @staticmethod
    def default_settings():
        return {
            "text_splitter": {
                "type": "sentence_splitter",
                "chunk_size": 800,
                "chunk_overlap": 400,
            },
            # https://platform.openai.com/docs/guides/embeddings/use-cases
            "vectorizer": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "batch_size": 100,
                "embedding_size": 256,
                "max_input_tokens": 8191,
            },
            "retriever_type": "dense_retriever",  # dense_retriever refers to embedding based retriever
            "retriever": {
                "provider": "faiss",
                "top_k": 5,
            },
        }

    def _load_documents(self, documents: List[Document]):
        self.documents = documents

    def _chunk_documents(self):
        self.chunks: List[Chunk] = []
        text_splitter = None
        text_splitter_settings = self.settings["text_splitter"]
        # TODO: wrap up text splitter into a class with __call__ method
        if text_splitter_settings["type"] == "sentence_splitter":
            text_splitter: SentenceSplitter = SentenceSplitter(
                chunk_size=text_splitter_settings["chunk_size"],
                chunk_overlap=text_splitter_settings["chunk_overlap"],
            )

        for doc in self.documents:
            chunks = text_splitter.split_text(doc.text)
            for i, chunk in enumerate(chunks):
                self.chunks.append(Chunk(vector=[], text=chunk, order=i, doc_id=doc.id))

    def _vectorize_chunks(self):
        """
        TODO: what happens when the text is None or too long
        """
        if not self.vectorizer:
            raise ValueError("Vectorizer is not set")
        batch_size = (
            50
            if "batch_size" not in self.settings["vectorizer"]
            else self.settings["vectorizer"]["batch_size"]
        )

        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i : i + batch_size]
            embedder_output: EmbedderOutput = self.vectorizer(
                [chunk.text for chunk in batch]
            )
            vectors = embedder_output.embeddings
            for j, vector in enumerate(vectors):
                self.chunks[i + j].vector = vector

            # update tracking
            self.tracking["vectorizer"]["num_calls"] += 1
            self.tracking["vectorizer"][
                "num_tokens"
            ] += embedder_output.usage.total_tokens

    def build_index(self, documents: List[Document]):
        self._load_documents(documents)
        self._chunk_documents()
        self._vectorize_chunks()
        self.retriever.set_chunks(self.chunks)
        print(
            f"Index built with {len(self.chunks)} chunks, {len(self.documents)} documents, ready for retrieval"
        )

    def retrieve(
        self, query_or_queries: Union[str, List[str]]
    ) -> Union[RetrieverOutput, List[RetrieverOutput]]:
        if not self.retriever:
            raise ValueError("Retriever is not set")
        retrieved = self.retriever(query_or_queries)
        if isinstance(query_or_queries, str):
            return retrieved[0] if retrieved else None
        return retrieved


if __name__ == "__main__":
    settings = {
        "text_splitter": {
            "type": "sentence_splitter",
            "chunk_size": 800,
            "chunk_overlap": 400,
        },
        "vectorizer": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "batch_size": 100,
            "embedding_size": 1536,
            "encoding_format": "float",  # from the default float64
        },
        "retriever_type": "dense_retriever",  # dense_retriever refers to embedding based retriever
        "retriever": {
            "provider": "faiss",
            "top_k": 5,
        },
    }
    doc1 = Document(
        meta_data={"title": "doc1"},
        text="This is a test document This is the second sentence " * 100,
        id="doc1",
    )
    doc2 = Document(
        meta_data={"title": "doc2"}, text="This is another test document", id="doc2"
    )
    rag = RAG(settings=settings)
    rag.build_index([doc1, doc2])
    print(rag.tracking)
    outputs = rag.retrieve(["test", "second sentence"])
    print(outputs)
