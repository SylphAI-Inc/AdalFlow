"""Demonstrating reranker to rank the documents and retrieve the top-k documents"""

from typing import List, Optional
import logging
from core.retriever import RetrieverOutputType, get_top_k_indices_scores
from lightrag.core.retriever import (
    Retriever,
    RetrieverOutput,
    RetrieverOutputType,
    RetrieverInputType,
)
from lightrag.components.model_client import TransformersClient, ModelType

log = logging.getLogger(__name__)


class RerankerRetriever(Retriever):
    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k
        self._model_name = "BAAI/bge-reranker-base"
        self.retriever = Retriever()
        self.model_client = TransformersClient(model_name=self._model_name)

    def reset_index(self):
        self.indexed = False
        self.documents = []

    def build_index_from_documents(self, documents: List[str]):
        self.documents = documents
        self.indexed = True

    def retrieve(
        self, input: RetrieverInputType, top_k: Optional[int] = None
    ) -> RetrieverOutputType:
        top_k = top_k or self.top_k
        queries = input if isinstance(input, List) else [input]
        retrieved_outputs: RetrieverOutputType = []

        for query in queries:
            input = [[query, doc] for doc in self.documents]
            api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
                input=input,
                model_kwargs={
                    "model": self._model_name,
                    "mock": False,
                },
                model_type=ModelType.RERANKER,
            )
            log.info(f"api_kwargs: {api_kwargs}")
            print(f"api_kwargs: {api_kwargs}    ")
            scores: List[float] = self.model_client.call(
                api_kwargs=api_kwargs, model_type=ModelType.RERANKER
            )
            top_k_indices, top_k_scores = get_top_k_indices_scores(scores, top_k)
            retrieved_outputs.append(
                RetrieverOutput(
                    doc_indices=top_k_indices,
                    doc_scores=top_k_scores,
                )
            )
        return retrieved_outputs
