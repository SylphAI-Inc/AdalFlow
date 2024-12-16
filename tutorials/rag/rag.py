from typing import Optional, Any, List

import adalflow as adal
from adalflow.core.db import LocalDB

from adalflow.core.types import ModelClientType

from adalflow.core.string_parser import JsonParser
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.components.data_process import (
    RetrieverOutputToContextStr,
    ToEmbeddings,
    TextSplitter,
)

from adalflow.components.model_client import OpenAIClient

from tutorials.rag.config import configs


def prepare_data_pipeline():
    splitter = TextSplitter(**configs["text_splitter"])
    embedder = adal.Embedder(
        model_client=ModelClientType.OPENAI(),
        model_kwargs=configs["embedder"]["model_kwargs"],
    )
    embedder_transformer = ToEmbeddings(
        embedder=embedder, batch_size=configs["embedder"]["batch_size"]
    )
    data_transformer = adal.Sequential(
        splitter, embedder_transformer
    )  # sequential will chain together splitter and embedder
    return data_transformer


rag_prompt_task_desc = r"""
You are a helpful assistant.

Your task is to answer the query that may or may not come with context information.
When context is provided, you should stick to the context and less on your prior knowledge to answer the query.

Output JSON format:
{
    "answer": "The answer to the query",
}"""


class RAG(adal.Component):

    def __init__(self, index_path: str = "index.faiss"):
        super().__init__()

        self.db = LocalDB.load_state(index_path)

        self.transformed_docs: List[adal.Document] = self.db.get_transformed_data(
            "data_transformer"
        )
        embedder = adal.Embedder(
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

        self.generator = adal.Generator(
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

        print(f"retrieved_documents: \n {retrieved_documents}\n")
        context_str = self.retriever_output_processors(retrieved_documents)

        print(f"context_str: \n {context_str}\n")

        return self.generate(query, context=context_str), retrieved_documents
