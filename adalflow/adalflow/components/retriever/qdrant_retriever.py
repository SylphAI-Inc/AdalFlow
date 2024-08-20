"""Leverage a Qdrant collection to retrieve documents."""

from typing import List, Optional, Any
from qdrant_client import QdrantClient, models

from adalflow.core.retriever import (
    Retriever,
)
from adalflow.core.embedder import Embedder

from adalflow.core.types import (
    RetrieverOutput,
    RetrieverStrQueryType,
    RetrieverStrQueriesType,
    Document,
)


class QdrantRetriever(Retriever[Any, RetrieverStrQueryType]):
    __doc__ = r"""Use a Qdrant collection to retrieve documents.

    Args:
        collection_name (str): the collection name in Qdrant.
        client (QdrantClient): An instance of qdrant_client.QdrantClient.
        embedder (Embedder): An instance of Embedder.
        top_k (Optional[int], optional): top k documents to fetch. Defaults to 10.
        vector_name (Optional[str], optional): the name of the vector in the collection. Defaults to None.
        text_key (str, optional): the key in the payload that contains the text. Defaults to "text".
        metadata_key (str, optional): the key in the payload that contains the metadata. Defaults to "meta_data".
        filter (Optional[models.Filter], optional): the filter to apply to the query. Defaults to None.

    References:
    [1] Qdrant: https://qdrant.tech/
    [2] Documentation: https://qdrant.tech/documentation/
    """

    def __init__(
        self,
        collection_name: str,
        client: QdrantClient,
        embedder: Embedder,
        top_k: Optional[int] = 10,
        vector_name: Optional[str] = None,
        text_key: str = "text",
        metadata_key: str = "meta_data",
        filter: Optional[models.Filter] = None,
    ):
        super().__init__()
        self._top_k = top_k
        self._collection_name = collection_name
        self._client = client
        self._embedder = embedder
        self._text_key = text_key
        self._metadata_key = metadata_key
        self._filter = filter

        self._vector_name = vector_name or self._get_first_vector_name()

    def reset_index(self):
        if self._client.collection_exists(self._collection_name):
            self._client.delete_collection(self._collection_name)

    def call(
        self,
        input: RetrieverStrQueriesType,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[RetrieverOutput]:
        top_k = top_k or self._top_k
        queries: List[str] = input if isinstance(input, list) else [input]

        queries_embeddings = self._embedder(queries)

        query_requests: List[models.QueryRequest] = []
        for idx, query in enumerate(queries):
            query_embedding = queries_embeddings.data[idx].embedding
            query_requests.append(
                models.QueryRequest(
                    query=query_embedding,
                    limit=top_k,
                    using=self._vector_name,
                    with_payload=True,
                    with_vector=True,
                    filter=self._filter,
                    **kwargs,
                )
            )

        results = self._client.query_batch_points(
            self._collection_name, requests=query_requests
        )
        retrieved_outputs: List[RetrieverOutput] = []
        for result in results:
            out = self._points_to_output(
                result.points,
                query,
                self._text_key,
                self._metadata_key,
                self._vector_name,
            )
            retrieved_outputs.append(out)

        return retrieved_outputs

    def _get_first_vector_name(self) -> Optional[str]:
        vectors = self._client.get_collection(
            self._collection_name
        ).config.params.vectors

        if not isinstance(vectors, dict):
            # The collection only has the default, unnamed vector
            return None

        first_vector_name = list(vectors.keys())[0]

        # The collection has multiple vectors. Could also include the falsy unnamed vector - Empty string("")
        return first_vector_name or None

    @classmethod
    def _points_to_output(
        cls,
        points: List[models.ScoredPoint],
        query: str,
        text_key: str,
        metadata_key: str,
        vector_name: Optional[str],
    ) -> RetrieverOutput:
        doc_indices = [point.id for point in points]
        doc_scores = [point.score for point in points]
        documents = [
            cls._doc_from_point(point, text_key, metadata_key, vector_name)
            for point in points
        ]
        return RetrieverOutput(
            doc_indices=doc_indices,
            doc_scores=doc_scores,
            query=query,
            documents=documents,
        )

    @classmethod
    def _doc_from_point(
        cls,
        point: models.ScoredPoint,
        text_key: str,
        metadata_key: str,
        vector_name: Optional[str] = None,
    ) -> Document:
        vector = point.vector
        if isinstance(vector, dict):
            vector = vector[vector_name]

        payload = point.payload.copy()
        return Document(
            id=point.id,
            text=payload.get(text_key, ""),
            meta_data=payload.get(metadata_key, {}),
            vector=vector,
        )
