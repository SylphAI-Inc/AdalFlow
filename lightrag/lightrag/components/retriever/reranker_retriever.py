"""Reranking model using modelclient as a retriever."""

from typing import List, Optional, Callable, Any, Dict
import logging

from lightrag.core.retriever import (
    Retriever,
)
from lightrag.core.types import (
    RetrieverStrQueriesType,
    RetrieverOutputType,
    RetrieverDocumentsType,
    RetrieverOutput,
    ModelType,
)
from lightrag.core.model_client import ModelClient

log = logging.getLogger(__name__)


class RerankerRetriever(Retriever[str, RetrieverStrQueriesType]):
    __doc__ = r"""
    A retriever that uses a reranker model to rank the documents and retrieve the top-k documents.

    Args:
        top_k (int, optional): The number of top documents to retrieve. Defaults to 5.
        model_client (ModelClient): The model client that has a reranker model,
            such as ``CohereAPIClient`` or ``TransformersClient``.
        model_kwargs (Dict): The model kwargs to pass to the model client.
        documents (Optional[RetrieverDocumentsType], optional): The documents to build the index from. Defaults to None.
        document_map_func (Optional[Callable[[Any], str]], optional): The function to map the document of Any type to the specific type ``RetrieverDocumentType`` that the retriever expects. Defaults to None.

    Examples:


    """

    def __init__(
        self,
        model_client: ModelClient,  # make sure you initialize the model client first
        model_kwargs: Dict = {},
        top_k: int = 5,
        documents: Optional[RetrieverDocumentsType] = None,
        document_map_func: Optional[Callable[[Any], str]] = None,
    ):
        super().__init__()
        self.top_k = top_k
        self._model_kwargs = model_kwargs or {}
        assert "model" in self._model_kwargs, "model must be specified in model_kwargs"

        self.model_client = model_client

        self.reset_index()
        if documents:
            self.build_index_from_documents(documents, document_map_func)

    def reset_index(self):
        self.indexed = False
        self.documents = []
        self.total_documents: int = 0

    def build_index_from_documents(
        self,
        documents: RetrieverDocumentsType,
        document_map_func: Optional[Callable[[Any], str]] = None,
    ):
        if document_map_func:
            documents = [document_map_func(doc) for doc in documents]
        else:
            documents = documents
        self.total_documents = len(documents)

        self._model_kwargs["documents"] = documents

        self.indexed = True

    def call(
        self, input: RetrieverStrQueriesType, top_k: Optional[int] = None
    ) -> RetrieverOutputType:
        top_k = top_k or self.top_k
        queries = input if isinstance(input, List) else [input]
        retrieved_outputs: RetrieverOutputType = []

        model_kwargs = self._model_kwargs.copy()
        model_kwargs["top_k"] = top_k

        for query in queries:
            api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
                input=query,
                model_kwargs=model_kwargs,
                model_type=ModelType.RERANKER,
            )
            log.info(f"api_kwargs: {api_kwargs}")
            top_k_indices, top_k_scores = self.model_client.call(
                api_kwargs=api_kwargs, model_type=ModelType.RERANKER
            )
            retrieved_outputs.append(
                RetrieverOutput(
                    doc_indices=top_k_indices,
                    doc_scores=top_k_scores,
                    query=query,
                )
            )
        return retrieved_outputs

    def _extra_repr(self) -> str:
        s = f"top_k={self.top_k}, model_kwargs={self._model_kwargs}, model_client={self.model_client}, total_documents={self.total_documents}"
        return s
