from typing import Any, List, Optional
import dotenv
import yaml

from lightrag.core.generator import Generator, GeneratorOutput
from lightrag.core.embedder import Embedder
from lightrag.core.data_components import (
    RetrieverOutputToContextStr,
    ToEmbeddings,
)
from lightrag.core.types import Document

from lightrag.core.document_splitter import DocumentSplitter
from lightrag.core.string_parser import JsonParser
from lightrag.core.component import Component, Sequential
from lightrag.core.db import LocalDocumentDB

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# from core.functional import generate_component_key

from lightrag.components.model_client import OpenAIClient
from lightrag.components.retriever import FAISSRetriever


dotenv.load_dotenv(dotenv_path=".env", override=True)


# TODO: RAG can potentially be a component itsefl and be provided to the users
class RAG(Component):

    def __init__(self, settings: dict):
        super().__init__()
        self.vectorizer_settings = settings["vectorizer"]
        self.retriever_settings = settings["retriever"]
        self.generator_model_kwargs = settings["generator"]
        self.text_splitter_settings = settings["text_splitter"]

        vectorizer = Embedder(
            model_client=OpenAIClient(),
            # batch_size=self.vectorizer_settings["batch_size"],
            model_kwargs=self.vectorizer_settings["model_kwargs"],
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
                vectorizer=vectorizer,
                batch_size=self.vectorizer_settings["batch_size"],
            ),
        )
        self.data_transformer_key = self.data_transformer._get_name()
        # initialize retriever, which depends on the vectorizer too
        self.retriever = FAISSRetriever(
            top_k=self.retriever_settings["top_k"],
            dimensions=self.vectorizer_settings["model_kwargs"]["dimensions"],
            vectorizer=vectorizer,
        )
        self.retriever_output_processors = RetrieverOutputToContextStr(deduplicate=True)
        # TODO: currently retriever will be applied on transformed data. but its not very obvious design pattern
        self.db = LocalDocumentDB(
            # retriever_transformer=data_transformer,  # prepare data for retriever to build index with
            # retriever=retriever,
            # retriever_output_processors=RetrieverOutputToContextStr(deduplicate=True),
        )

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
        self.tracking = {"vectorizer": {"num_calls": 0, "num_tokens": 0}}

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
        if response.error:
            raise ValueError(f"Error in generator: {response.error}")
        return response.data

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

        return self.generate(query, context=context_str), context_str


if __name__ == "__main__":
    with open("./configs/rag.yaml", "r") as file:
        settings = yaml.safe_load(file)
    print(settings)
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
    rag = RAG(settings)
    print(rag)
    rag.build_index([doc1, doc2])
    print(rag.tracking)
    query = "What is Li Yin's hobby and profession?"

    response, _ = rag.call(query)

    print(f"execution graph: {rag._execution_graph}")
    print(f"response: {response}")
    print(f"subcomponents: {rag._components}")
    rag.visualize_graph_html("my_component_graph.html")
