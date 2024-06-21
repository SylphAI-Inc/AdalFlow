from typing import Any, List, Optional


from lightrag.core.generator import Generator
from lightrag.core.embedder import Embedder

from lightrag.core.types import Document
from lightrag.core.string_parser import JsonParser
from lightrag.core.component import Component, Sequential
from lightrag.core.db import LocalDB

from lightrag.components.retriever import FAISSRetriever
from lightrag.components.model_client import OpenAIClient

from lightrag.components.data_process import (
    RetrieverOutputToContextStr,
    ToEmbeddings,
    DocumentSplitter,
)
from lightrag.utils import setup_env  # noqa


# TODO: RAG can potentially be a component itsefl and be provided to the users
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

        vectorizer = Embedder(
            model_client=OpenAIClient(),
            # batch_size=self.vectorizer_settings["batch_size"], #TODO: where to put the batch size control and how big can it go?
            model_kwargs=self.vectorizer_settings["model_kwargs"],
            # output_processors=ToEmbedderResponse(),
        )
        # TODO: check document splitter, how to process the parent and order of the chunks
        text_splitter = DocumentSplitter(
            split_by=self.text_splitter_settings["split_by"],
            split_length=self.text_splitter_settings["chunk_size"],
            split_overlap=self.text_splitter_settings["chunk_overlap"],
        )
        self.data_transformer = Sequential(
            text_splitter,
            ToEmbeddings(
                embedder=vectorizer,
                batch_size=self.vectorizer_settings["batch_size"],
            ),
        )
        # TODO: make a new key
        self.data_transformer_key = self.data_transformer._get_name()
        # initialize retriever, which depends on the vectorizer too
        self.retriever = FAISSRetriever(
            top_k=self.retriever_settings["top_k"],
            dimensions=self.vectorizer_settings["model_kwargs"]["dimensions"],
            vectorizer=vectorizer,
        )
        self.retriever_output_processors = RetrieverOutputToContextStr(deduplicate=True)
        # TODO: currently retriever will be applied on transformed data. but its not very obvious design pattern
        self.db = LocalDB()

        # initialize generator
        self.generator = Generator(
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
            model_client=OpenAIClient(),
            model_kwargs=self.generator_model_kwargs,
            output_processors=JsonParser(),
        )
        self.tracking = {
            "vectorizer": {"num_calls": 0, "num_tokens": 0}
        }  # TODO: tracking of the usage can be added in default in APIClient component

    def build_index(self, documents: List[Document]):
        self.db.load_documents(documents)
        self.map_key = self.db.map_data()
        print(f"map_key: {self.map_key}")
        self.data_key = self.db.transform_data(self.data_transformer)
        print(f"data_key: {self.data_key}")
        self.transformed_documents = self.db.get_transformed_data(self.data_key)
        self.retriever.build_index_from_documents(self.transformed_documents)

    def generate(self, query: str, context: Optional[str] = None) -> Any:
        if not self.generator:
            raise ValueError("Generator is not set")

        prompt_kwargs = {
            "context_str": context,
            "input_str": query,
        }
        response = self.generator(prompt_kwargs=prompt_kwargs)
        return response

    def call(self, query: str) -> Any:
        retrieved_documents = self.retriever(query)
        # fill in the document
        for i, retriever_output in enumerate(retrieved_documents):
            retrieved_documents[i].documents = [
                self.transformed_documents[doc_index]
                for doc_index in retriever_output.doc_indexes
            ]
        # convert all the documents to context string

        context_str = self.retriever_output_processors(retrieved_documents)

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

    # print(f"execution graph: {rag._execution_graph}")
    print(f"response: {response}")
    # print(f"subcomponents: {rag._components}")
    # rag.visualize_graph_html("my_component_graph.html")
