"""Demonstrating reranker to rank the documents and retrieve the top-k documents"""

from typing import List, Optional, Callable, Any
import logging

from lightrag.core.retriever import (
    Retriever,
    get_top_k_indices_scores,
)
from lightrag.core.types import (
    RetrieverInputStrType,
    RetrieverOutputType,
    RetrieverDocumentsType,
    RetrieverOutput,
    ModelType,
)
from lightrag.components.model_client import TransformersClient

log = logging.getLogger(__name__)


class RerankerRetriever(Retriever[str, RetrieverInputStrType]):
    r"""
    A retriever that uses a reranker model to rank the documents and retrieve the top-k documents.

    Args:
        top_k (int, optional): The number of top documents to retrieve. Defaults to 5.
    """

    def __init__(
        self,
        top_k: int = 5,
        documents: Optional[RetrieverDocumentsType] = None,
        document_map_func: Optional[Callable[[Any], str]] = None,
    ):
        super().__init__()
        self.top_k = top_k
        self._model_name = "BAAI/bge-reranker-base"
        self.model_client = TransformersClient(model_name=self._model_name)
        if documents:
            self.build_index_from_documents(documents, document_map_func)

    def reset_index(self):
        self.indexed = False
        self.documents = []

    def build_index_from_documents(
        self,
        documents: RetrieverDocumentsType,
        document_map_func: Optional[Callable[[Any], str]] = None,
    ):
        if document_map_func:
            documents = [document_map_func(doc) for doc in documents]
        else:
            documents = documents
        self.indexed = True

    def retrieve(
        self, input: RetrieverInputStrType, top_k: Optional[int] = None
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
                },
                model_type=ModelType.RERANKER,
            )
            log.info(f"api_kwargs: {api_kwargs}")
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
