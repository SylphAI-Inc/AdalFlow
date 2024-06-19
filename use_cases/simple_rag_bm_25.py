from typing import Any, List, Optional

from core.generator import Generator
from core.data_components import (
    RetrieverOutputToContextStr,
)

from core.string_parser import JsonParser
from core.component import Component, Sequential
from core.db import LocalDB
from core.types import Document

from components.retriever import InMemoryBM25Retriever
from components.model_client import OpenAIClient

import utils.setup_env  # noqa


# TODO: RAG can potentially be a component itsefl and be provided to the users
class RAG(Component):

    def __init__(self):
        super().__init__()

        self.retriever_settings = {
            "top_k": 1,
        }
        self.generator_model_kwargs = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.3,
            "stream": False,
        }

        self.retriever = InMemoryBM25Retriever(
            top_k=self.retriever_settings["top_k"],
        )
        self.retriever_output_processors = RetrieverOutputToContextStr(deduplicate=True)

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
            model_client=OpenAIClient,
            model_kwargs=self.generator_model_kwargs,
            output_processors=Sequential(JsonParser()),
        )
        self.tracking = {"vectorizer": {"num_calls": 0, "num_tokens": 0}}

    def build_index(self, documents: List[Document]):
        self.db.load_documents(documents)
        # self.db()  # transform the documents
        self.retriever.build_index_from_documents(documents=self.db.documents)
        # self.db.build_retrieve_index()

    def generate(self, query: str, context: Optional[str] = None) -> Any:
        if not self.generator:
            raise ValueError("Generator is not set")

        prompt_kwargs = {
            "context_str": context,
        }
        response = self.generator.call(input=query, prompt_kwargs=prompt_kwargs)
        return response

    def call(self, query: str) -> Any:
        retrieved_documents = self.retriever(query)
        # fill in the document content
        for i, retriever_output in enumerate(retrieved_documents):
            retrieved_documents[i].documents = [
                self.transformed_documents[doc_id]
                for doc_id in retriever_output.doc_indexes
            ]
        # apply the retriever output processors
        context_str = self.retriever_output_processors(retrieved_documents)
        return self.generate(query, context=context_str)


if __name__ == "__main__":
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
    query = "What is Li Yin's profession?"

    response = rag.call(query)
    print(f"response: {response}")
