from typing import Any, List, Optional
import os
from adalflow.core import Component, Generator, Embedder, Sequential
from adalflow.core.types import Document, ModelClientType
from adalflow.core.string_parser import JsonParser
from adalflow.core.db import LocalDB
from adalflow.utils import setup_env

from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.components.model_client import OpenAIClient

from adalflow.components.data_process import (
    RetrieverOutputToContextStr,
    ToEmbeddings,
    TextSplitter,
)

setup_env()
# TODO: RAG can potentially be a component itsefl and be provided to the users

configs = {
    "embedder": {
        "batch_size": 100,
        "model_kwargs": {
            "model": "text-embedding-3-small",
            "dimensions": 256,
            "encoding_format": "float",
        },
    },
    "retriever": {
        "top_k": 2,
    },
    "generator": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.3,
        "stream": False,
    },
    "text_splitter": {
        "split_by": "word",
        "chunk_size": 400,
        "chunk_overlap": 200,
    },
}


# use data process complete that will transform on Document structure
def prepare_data_pipeline():
    splitter = TextSplitter(**configs["text_splitter"])
    embedder = Embedder(
        model_client=ModelClientType.OPENAI(),
        model_kwargs=configs["embedder"]["model_kwargs"],
    )
    embedder_transformer = ToEmbeddings(
        embedder=embedder, batch_size=configs["embedder"]["batch_size"]
    )
    data_transformer = Sequential(splitter, embedder_transformer)
    return data_transformer


def prepare_database_with_index(docs: List[Document], index_path: str = "index.faiss"):
    if os.path.exists(index_path):
        return None
    db = LocalDB()
    db.load(docs)
    data_transformer = prepare_data_pipeline()
    db.transform(data_transformer, key="data_transformer")
    # store
    db.save_state(index_path)


rag_prompt_task_desc = r"""
You are a helpful assistant.

Your task is to answer the query that may or may not come with context information.
When context is provided, you should stick to the context and less on your prior knowledge to answer the query.

Output JSON format:
{
    "answer": "The answer to the query",
}"""


class RAG(Component):

    def __init__(self, index_path: str = "index.faiss"):
        super().__init__()

        self.db = LocalDB.load_state(index_path)

        self.transformed_docs: List[Document] = self.db.get_transformed_data(
            "data_transformer"
        )
        embedder = Embedder(
            model_client=ModelClientType.OPENAI(),
            model_kwargs=configs["embedder"]["model_kwargs"],
        )
        # map the documents to embeddings
        self.retriever = FAISSRetriever(
            **configs["retriever"],
            embedder=embedder,
            documents=self.transformed_docs,
            document_map_func=lambda doc: doc.vector,
        )
        self.retriever_output_processors = RetrieverOutputToContextStr(deduplicate=True)
        self.generator = Generator(
            model_client=ModelClientType.OPENAI(),
            model_kwargs=configs["generator"],
            output_processors=JsonParser(),
        )

        self.generator = Generator(
            prompt_kwargs={
                "task_desc_str": rag_prompt_task_desc,
            },
            model_client=OpenAIClient(),
            model_kwargs=configs["generator"],
            output_processors=JsonParser(),
        )

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
                self.transformed_docs[doc_index]
                for doc_index in retriever_output.doc_indices
            ]

        print(f"retrieved_documents: \n {retrieved_documents}")
        context_str = self.retriever_output_processors(retrieved_documents)

        print(f"context_str: \n {context_str}")

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
        + "Li Yin is an AI researcher and a software engineer"
        + "lots of more nonsense text" * 250,
        id="doc2",
    )
    # only run it once to prepare the data, if index exists, it will not run
    prepare_database_with_index([doc1, doc2], index_path="index.faiss")
    rag = RAG(index_path="index.faiss")
    print(rag)
    query = "What is Li Yin's hobby and profession?"

    response = rag.call(query)

    print(f"response: {response}")
