from typing import Any, Dict, List, Union, Optional

from core.component import (
    OpenAIEmbedder,
    Component,
    RetrieverOutput,
    FAISSRetriever,
    EmbedderOutput,
)
from core.openai_llm import OpenAIGenerator
from core.light_rag import DEFAULT_QA_PROMPT
from core.data_classes import Document, Chunk

# TODO: rewrite SentenceSplitter and other splitter classes
from llama_index.core.node_parser import SentenceSplitter
from jinja2 import Template
from core.component import Component


class Query(Component):
    """
    Allow to define query.
    #TODO: support batch processing
    """

    def __init__(self, query: str, session_id: str = None):
        self.query = query
        self.session_id = session_id


from core.prompt_builder import Prompt


##############################################
# Main RAG class
# Configs and combines functional modules
# One settings per RAG instance instead of global settings
##############################################
class RAG(Component):
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
        # initialize generator
        self.generator = None
        if "generator" in self.settings:
            generator_settings = self.settings["generator"]
            if generator_settings["provider"] == "openai":
                self.generator = OpenAIGenerator(
                    # provider=generator_settings["provider"],
                    # model=generator_settings["model"],
                    **generator_settings,
                )

        self.tracking = {"vectorizer": {"num_calls": 0, "num_tokens": 0}}
        self.llm_task_desc = """
You are a helpful assistant.

Your task is to answer the query that may or may not come with context information.
When context is provided, you should stick to the context and less on your prior knowledge to answer the query.
    """

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
            "generator": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "temperature": 0.3,
                "stream": False,
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

        # system_prompt_template = Template(DEFAULT_QA_PROMPT)
        # context_str = context if context else ""
        # query_str = query
        # system_prompt_content = system_prompt_template.render(
        #     context_str=context_str, query_str=query_str
        # )
        # messages = [
        #     {"role": "system", "content": system_prompt_content},
        # ]
        # print(f"messages: {messages}")
        response = self.generator.call(
            input=query, context_str=context, task_desc_str=self.llm_task_desc
        )
        return response


if __name__ == "__main__":
    # NOTE: for the ouput of this following code, check text_lightrag.txt
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
            "embedding_size": 256,
            "encoding_format": "float",  # from the default float64
        },
        "retriever_type": "dense_retriever",  # dense_retriever refers to embedding based retriever
        "retriever": {
            "provider": "faiss",
            "top_k": 1,
        },
        "generator": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.3,
            "stream": False,
        },
    }
    doc1 = Document(
        meta_data={"title": "Li Yin's profile"},
        text="My name is Li Yin, I love rock climbing" + "lots of nonsense text" * 1000,
        id="doc1",
    )
    doc2 = Document(
        meta_data={"title": "Interviewing Li Yin"},
        text="lots of more nonsense text" * 500
        + "Li Yin is a software developer and AI researcher"
        + "lots of more nonsense text" * 500,
        id="doc2",
    )
    rag = RAG(settings=settings)
    rag.build_index([doc1, doc2])
    print(rag.tracking)
    query = "What is Li Yin's hobby and profession?"
    # in this simple case, query expansion is not necessary, this is only for demonstration the list input of queries
    # the anaswer will only be right if each expended query has the right relevant chunk as set top_k to 1
    expanded_queries = ["Li Yin's hobby", "Li Yin's profession"]
    outputs = rag.retrieve(expanded_queries)
    print(f"retrieved: {outputs}")
    context_str = rag.retriever_output_to_context_str(outputs)
    response = rag.generate(query, context_str)
    print(f"response: {response}")
    # now try to set top_k to 2, and see if the answer is still correct
    # or set chunk_size to 20, chunk_overlap to 10, and see if the answer is still correct
    # you can try to fit all the context into the prompt, use long-context LLM.
