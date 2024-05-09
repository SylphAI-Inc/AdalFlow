from typing import Any, Dict, List, Union, Optional


# from core.openai_llm import OpenAIGenerator
from core.openai_client import OpenAIClient
from core.generator import Generator
from core.embedder import Embedder
from core.documents_data_class import Document, Chunk
from core.data_classes import EmbedderResponse
from core.data_components import ToEmbedderResponse

# TODO: rewrite SentenceSplitter and other splitter classes
# from llama_index.core.node_parser import SentenceSplitter
from core.document_splitter import DocumentSplitter
from core.component import Component, Sequential
from core.string_parser import JsonParser
from core.component import (
    Component,
    RetrieverOutput,
)
from core.retriever import FAISSRetriever
from core.db import LocalDocumentDB
from core.document_splitter import DocumentSplitter
from core.data_components import ToEmbeddings
import dotenv

dotenv.load_dotenv(dotenv_path=".env", override=True)


class RAG(Component):

    def __init__(self):
        super().__init__()

        self.vectorizer_settings = {
            "batch_size": 100,
            "model_kwargs": {
                "model": "text-embedding-3-small",
                "dimensions": 256,
                "encoding_format": "float",
            },
        }
        self.retriever_settings = {
            "top_k": 2,
        }
        self.generator_model_kwargs = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.3,
            "stream": False,
        }
        self.text_splitter_settings = {  # TODO: change it to direct to spliter kwargs
            "split_by": "word",
            "chunk_size": 400,
            "chunk_overlap": 200,
        }

        self.vectorizer = Embedder(
            model_client=OpenAIClient(),
            # batch_size=self.vectorizer_settings["batch_size"],
            model_kwargs=self.vectorizer_settings["model_kwargs"],
            output_processors=Sequential(ToEmbedderResponse()),
        )
        text_splitter = DocumentSplitter(
            split_by=self.text_splitter_settings["split_by"],
            split_length=self.text_splitter_settings["chunk_size"],
            split_overlap=self.text_splitter_settings["chunk_overlap"],
        )
        data_transformer = Sequential(
            text_splitter,
            ToEmbeddings(
                vectorizer=self.vectorizer,
                batch_size=self.vectorizer_settings["batch_size"],
            ),
        )
        self.db = LocalDocumentDB(data_transformer=data_transformer)
        # initialize retriever, which depends on the vectorizer too
        self.retriever = FAISSRetriever(
            top_k=self.retriever_settings["top_k"],
            d=self.vectorizer_settings["model_kwargs"]["dimensions"],
            vectorizer=self.vectorizer,
            # db=self.db,
        )
        # initialize generator
        self.generator = Generator(
            model_client=OpenAIClient(),
            output_processors=Sequential(JsonParser()),
            preset_prompt_kwargs={
                "task_desc_str": r"""
You are a helpful assistant.

Your task is to answer the query that may or may not come with context information.
When context is provided, you should stick to the context and less on your prior knowledge to answer the query.

Output JSON format:
{
    "answer": "The answer to the query",
}"""
            },
            model_kwargs=self.generator_model_kwargs,
        )
        self.tracking = {"vectorizer": {"num_calls": 0, "num_tokens": 0}}

    # def _load_documents(self, documents: List[Document]):
    #     self.documents = documents

    # def _chunk_documents(self):
    #     self.chunks: List[Chunk] = []
    #     text_splitter: DocumentSplitter = DocumentSplitter(
    #         split_by=self.text_splitter_settings["split_by"],
    #         split_length=self.text_splitter_settings["chunk_size"],
    #         split_overlap=self.text_splitter_settings["chunk_overlap"],
    #     )
    #     self.chunks = text_splitter(self.documents)
    #     print(f"Chunked {len(self.documents)} documents into {len(self.chunks)} chunks")

    # def _vectorize_chunks(self):
    #     """
    #     TODO: what happens when the text is None or too long
    #     """
    #     if not self.vectorizer:
    #         raise ValueError("Vectorizer is not set")
    #     batch_size = (
    #         50
    #         if "batch_size" not in self.vectorizer_settings
    #         else self.vectorizer_settings["batch_size"]
    #     )

    #     for i in range(0, len(self.chunks), batch_size):
    #         batch = self.chunks[i : i + batch_size]
    #         embedder_output: EmbedderResponse = self.vectorizer(
    #             input=[chunk.text for chunk in batch]
    #         )
    #         vectors = embedder_output.data
    #         for j, vector in enumerate(vectors):
    #             self.chunks[i + j].vector = vector.embedding

    #         # update tracking
    #         self.tracking["vectorizer"]["num_calls"] += 1
    #         self.tracking["vectorizer"][
    #             "num_tokens"
    #         ] += embedder_output.usage.total_tokens

    def build_index(self, documents: List[Document]):
        self.db.load_documents(documents)
        self.db()  # transform the documents
        self.retriever.set_chunks(self.db.transformed_documents)
        # self._load_documents(documents)
        # self._chunk_documents()
        # self._vectorize_chunks()
        # self.retriever.set_chunks(self.chunks)
        # print(
        #     f"Index built with {len(self.chunks)} chunks, {len(self.documents)} documents, ready for retrieval"
        # )

    def retrieve(
        self, query_or_queries: Union[str, List[str]]
    ) -> Union[RetrieverOutput, List[RetrieverOutput]]:
        if not self.retriever:
            raise ValueError("Retriever is not set")
        retrieved = self.retriever(query_or_queries)
        if isinstance(query_or_queries, str):
            return retrieved[0] if retrieved else None
        return retrieved

    @staticmethod
    def retriever_output_to_context_str(
        retriever_output: Union[RetrieverOutput, List[RetrieverOutput]],
        deduplicate: bool = False,
    ) -> str:
        """
        How to combine your retrieved chunks into the context is highly dependent on your use case.
        If you used query expansion, you might want to deduplicate the chunks.
        """
        chunks_to_use: List[Chunk] = []
        context_str = ""
        sep = " "
        if isinstance(retriever_output, RetrieverOutput):
            chunks_to_use = retriever_output.chunks
        else:
            for output in retriever_output:
                chunks_to_use.extend(output.chunks)
        if deduplicate:
            unique_chunks_ids = set([chunk.id for chunk in chunks_to_use])
            # id and if it is used, it will be True
            used_chunk_in_context_str: Dict[Any, bool] = {
                id: False for id in unique_chunks_ids
            }
            for chunk in chunks_to_use:
                if not used_chunk_in_context_str[chunk.id]:
                    context_str += sep + chunk.text
                    used_chunk_in_context_str[chunk.id] = True
        else:
            context_str = sep.join([chunk.text for chunk in chunks_to_use])
        return context_str

    def generate(self, query: str, context: Optional[str] = None) -> Any:
        if not self.generator:
            raise ValueError("Generator is not set")

        prompt_kwargs = {
            "context_str": context,
        }
        response = self.generator.call(input=query, prompt_kwargs=prompt_kwargs)
        return response

    def call(self, query: str) -> Any:
        retriever_output = self.retrieve(query)
        context_str = self.retriever_output_to_context_str(retriever_output)

        return self.generate(query, context=context_str)


if __name__ == "__main__":
    # NOTE: for the ouput of this following code, check text_lightrag.txt
    doc1 = Document(
        meta_data={"title": "Li Yin's profile"},
        text="My name is Li Yin, I love rock climbing" + "lots of nonsense text" * 500,
        id="doc1",
    )
    doc2 = Document(
        meta_data={"title": "Interviewing Li Yin"},
        text="lots of more nonsense text" * 250
        + "Li Yin is a software developer and AI researcher"
        + "lots of more nonsense text" * 250,
        id="doc2",
    )
    rag = RAG()
    print(rag)
    rag.build_index([doc1, doc2])
    print(rag.tracking)
    query = "What is Li Yin's hobby and profession?"

    response = rag.call(query)
    print(f"response: {response}")
