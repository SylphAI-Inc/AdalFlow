from typing import List

from lightrag.core.types import Document
from lightrag.core.retriever import Retriever
from lightrag.core.embedder import Embedder
from lightrag.components.data_process import (
    RetrieverOutputToContextStr,
    ToEmbeddings,
    DocumentSplitter,
)
from lightrag.core.db import LocalDB
from lightrag.core.component import Component, Sequential


class RetrievalICL(Component):
    def __init__(
        self,
        retriever: Retriever,
        retriever_output_processors: RetrieverOutputToContextStr,
        text_splitter: DocumentSplitter,
        vectorizer: Embedder,
        db: LocalDB,
    ):
        super().__init__()
        self.retriever = retriever
        self.retriever_output_processors = retriever_output_processors

        self.text_splitter = text_splitter
        self.vectorizer = vectorizer
        self.data_transformer = Sequential(
            self.text_splitter,
            ToEmbeddings(
                embedder=self.vectorizer,
            ),
        )
        self.data_transformer_key = self.data_transformer._get_name()
        self.db = db

    def build_index(self, documents: List[Document]):
        self.db.load_documents(documents)
        self.map_key = self.db.map_data()
        print(f"map_key: {self.map_key}")
        self.data_key = self.db.transform_data(self.data_transformer)
        print(f"data_key: {self.data_key}")
        self.transformed_documents = self.db.get_transformed_data(self.data_key)
        self.retriever.build_index_from_documents(self.transformed_documents)

    def call(self, query: str, top_k: int) -> str:
        retrieved_documents = self.retriever(query, top_k)
        # fill in the document
        for i, retriever_output in enumerate(retrieved_documents):
            retrieved_documents[i].documents = [
                self.transformed_documents[doc_index]
                for doc_index in retriever_output.doc_indexes
            ]
        example_str = self.retriever_output_processors(retrieved_documents)
        return example_str
