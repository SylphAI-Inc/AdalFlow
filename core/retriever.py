from copy import deepcopy
from typing import List, Optional, Union, Any

import faiss
import numpy as np

from core.component import Component
from core.data_classes import (
    Chunk,
    Document,
    RetrieverOutput,
)

RetrieverInputType = Union[str, List[str]]
RetrieverOutputType = List[RetrieverOutput]


class Retriever(Component):
    """
    Retriever will manage its own index and retrieve in format of RetrieverOutput
    It does not manage the initial documents.
    """

    indexed = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_index(self):
        raise NotImplementedError(f"reset_index is not implemented")

    def build_index_from_documents(self, documents: List[Document]):
        raise NotImplementedError(f"build_index_from_documents is not implemented")

    def retrieve(
        self, query_or_queries: RetrieverInputType, top_k: Optional[int] = None
    ) -> RetrieverOutputType:
        raise NotImplementedError(f"retrieve is not implemented")

    def __call__(
        self, query_or_queries: RetrieverInputType, top_k: Optional[int] = None
    ) -> RetrieverOutputType:
        raise NotImplementedError(f"__call__ is not implemented")


class FAISSRetriever(Component):
    """
    https://github.com/facebookresearch/faiss
    To use the retriever,
    (1) build index from Document chunks
    (2) query the retriever with a query string

    The retriever uses in-memory Faiss index to retrieve the top k chunks
    d: dimension of the vectors
    xb: number of vectors to put in the index
    xq: number of queries
    The data type dtype must be float32.
    Note: When the num of chunks are less than top_k, the last columns will be -1

    Other index options:
    - faiss.IndexFlatL2: L2 or Euclidean distance, [-inf, inf]
    - faiss.IndexFlatIP: inner product of normalized vectors will be cosine similarity, [-1, 1]

    We choose cosine similarity and convert it to range [0, 1] by adding 1 and dividing by 2 to simulate probability
    """

    name = "FAISSRetriever"

    def __init__(
        self,
        *,
        # arguments
        top_k: int = 3,
        dimensions: int = 768,
        chunks: Optional[List[Chunk]] = None,
        # components
        vectorizer: Optional[Component] = None,
        document_db: Optional[Component] = None,
        output_processors: Optional[Component] = None,
    ):
        super().__init__(provider="Meta")
        self.dimensions = dimensions
        self.index = faiss.IndexFlatIP(
            dimensions
        )  # inner product of normalized vectors will be cosine similarity, [-1, 1]

        self.vectorizer = vectorizer  # used to vectorize the queries
        if chunks:
            self.set_chunks(chunks)
        else:
            self.chunks: List[Chunk] = []
            self.total_chunks: int = 0
        self.top_k = top_k
        self.document_db = document_db  # it can directly use the data from db or directly from the chunks
        self.output_processors = output_processors

    def reset(self):
        self.index.reset()
        self.chunks: List[Chunk] = []
        self.total_chunks: int = 0

    def set_chunks(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.total_chunks = len(chunks)
        embeddings = [chunk.vector for chunk in chunks]
        xb = np.array(embeddings, dtype=np.float32)
        self.index.add(xb)

    def load_index(self, chunks: List[Chunk]):
        """
        Ensure embeddings are already in the chunks
        """
        self.set_chunks(chunks)

    def _convert_cosine_similarity_to_probability(self, D: np.ndarray) -> np.ndarray:
        D = (D + 1) / 2
        D = np.round(D, 3)
        return D

    def _to_retriever_output(
        self, Ind: np.ndarray, D: np.ndarray
    ) -> List[RetrieverOutput]:
        output: List[RetrieverOutput] = []
        # Step 1: Filter out the -1, -1 columns along with its scores when top_k > len(chunks)
        if -1 in Ind:
            valid_columns = ~np.any(Ind == -1, axis=0)

            D = D[:, valid_columns]
            Ind = Ind[:, valid_columns]
        # Step 2: processing rows (one query at a time)
        for row in zip(Ind, D):
            indexes, distances = row
            chunks: List[Chunk] = []
            for index, distance in zip(indexes, distances):
                chunk: Chunk = deepcopy(self.chunks[index])
                chunk.score = distance
                chunks.append(chunk)

            output.append(RetrieverOutput(chunks=chunks))

        return output

    def retrieve(
        self, query_or_queries: Union[str, List[str]], top_k: Optional[int] = None
    ) -> List[RetrieverOutput]:
        # if you pass a single query, you should access the first element of the list
        if self.index.ntotal == 0:
            raise ValueError(
                "Index is empty. Please set the chunks to build the index from"
            )
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [q for q in queries if q]  # Filter empty queries
        queries_embeddings = self.vectorizer(input=queries).data
        queries_embeddings = [data.embedding for data in queries_embeddings]
        xq = np.array(queries_embeddings, dtype=np.float32)
        D, Ind = self.index.search(xq, top_k if top_k else self.top_k)
        D = self._convert_cosine_similarity_to_probability(D)
        retrieved_output = self._to_retriever_output(Ind, D)
        for i, output in enumerate(retrieved_output):
            output.query = queries[i]
        return retrieved_output

    def __call__(
        self, query_or_queries: Union[str, List[str]], top_k: Optional[int] = None
    ) -> Any:
        response = self.retrieve(query_or_queries=query_or_queries, top_k=top_k)
        if self.output_processors:
            response = self.output_processors(response)
        return response

    def extra_repr(self) -> str:
        s = f"top_k={self.top_k}, dimensions={self.dimensions}, total_chunks={len(self.chunks)}, "
        return s
