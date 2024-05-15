from typing import List, Optional, Union, Callable, Any, Sequence, Any, overload

import faiss
import numpy as np

from core.component import Component
from core.data_classes import (
    Document,
    RetrieverOutput,
)

RetrieverInputType = Union[str, List[str]]  # query
RetrieverDocumentType = Any  # Documents
RetrieverOutputType = List[RetrieverOutput]


class Retriever(Component):
    """
    Retriever will manage its own index and retrieve in format of RetrieverOutput
    It takes a list of Document and builds index used for retrieval use anyway formed content from fields of the document
    using the input_field_map_func
    """

    indexed = False

    def __init__(self, *args, **kwargs):
        super().__init__()

    def reset_index(self):
        raise NotImplementedError(f"reset_index is not implemented")

    def _get_inputs(self, documents: List[Any], input_field_map_func: Callable):
        return [input_field_map_func(document) for document in documents]

    def build_index_from_documents(
        self,
        documents: RetrieverDocumentType,
        input_field_map_func: Callable[[Any], Any] = lambda x: x.text,
    ):
        r"""Built index from the `text` field of each document in the list of documents.
        input_field_map_func: a function that maps the document to the input field to be used for indexing
        You can use _get_inputs to get a standard format fits for this retriever or you can write your own
        """
        raise NotImplementedError(
            f"build_index_from_documents and input_field_map_func is not implemented"
        )

    def retrieve(
        self, query_or_queries: RetrieverInputType, top_k: Optional[int] = None
    ) -> Any:
        raise NotImplementedError(f"retrieve is not implemented")

    def __call__(
        self, query_or_queries: RetrieverInputType, top_k: Optional[int] = None
    ) -> Any:
        raise NotImplementedError(f"__call__ is not implemented")


FAISSRetrieverDocumentType = Sequence[List[float]]  # embeddings


class FAISSRetriever(Retriever):
    """
    Ensure the documents/chunks already have embeddings
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

    def __init__(
        self,
        *,
        # arguments
        top_k: int = 3,
        dimensions: int = 768,
        # components
        vectorizer: Optional[Component] = None,
    ):
        super().__init__()
        self.dimensions = dimensions
        self.index = faiss.IndexFlatIP(
            dimensions
        )  # inner product of normalized vectors will be cosine similarity, [-1, 1]

        self.vectorizer = vectorizer  # used to vectorize the queries

        self.top_k = top_k

    def reset_index(self):
        self.index.reset()
        self.total_chunks: int = 0
        self.indexed = False

    # TODO: Callable or AsyncCallable
    def build_index_from_documents(
        self,
        documents: FAISSRetrieverDocumentType,
        # input_field_map_func: Callable[[Any], Sequence[float]] = lambda x: x.vector,
    ):
        r"""Built index from the `vector` field of each document in the list of documents"""
        self.total_chunks = len(documents)
        # embeddings = [input_field_map_func(document) for document in documents]
        embeddings = [document.vector for document in documents]
        # embeddings = self._get_inputs(documents, input_field_map_func)
        print(f"embeddings: {embeddings}")
        xb = np.array(embeddings, dtype=np.float32)
        self.index.add(xb)
        self.indexed = True

    # def set_chunks(self, chunks: List[Chunk]):
    #     self.chunks = chunks
    #     self.total_chunks = len(chunks)
    #     embeddings = [chunk.vector for chunk in chunks]
    #     xb = np.array(embeddings, dtype=np.float32)
    #     self.index.add(xb)

    # def load_index(self, chunks: List[Chunk]):
    #     """
    #     Ensure embeddings are already in the chunks
    #     """
    #     self.set_chunks(chunks)

    def _convert_cosine_similarity_to_probability(self, D: np.ndarray) -> np.ndarray:
        D = (D + 1) / 2
        D = np.round(D, 3)
        return D

    def _to_retriever_output(
        self, Ind: np.ndarray, D: np.ndarray
    ) -> RetrieverOutputType:
        output: RetrieverOutputType = []
        # Step 1: Filter out the -1, -1 columns along with its scores when top_k > len(chunks)
        if -1 in Ind:
            valid_columns = ~np.any(Ind == -1, axis=0)

            D = D[:, valid_columns]
            Ind = Ind[:, valid_columns]
        # Step 2: processing rows (one query at a time)
        for row in zip(Ind, D):
            indexes, distances = row
            # chunks: List[Chunk] = []
            retrieved_documents_indexes = indexes
            retrieved_documents_scores = distances
            output.append(
                RetrieverOutput(
                    doc_indexes=retrieved_documents_indexes,
                    doc_scores=retrieved_documents_scores,
                )
            )

        return output

    def retrieve(
        self, query_or_queries: Union[str, List[str]], top_k: Optional[int] = None
    ) -> RetrieverOutputType:
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
        # TODO: improve the embedding output format
        queries_embeddings = self.vectorizer(input=queries).data
        queries_embeddings = [data.embedding for data in queries_embeddings]
        xq = np.array(queries_embeddings, dtype=np.float32)
        D, Ind = self.index.search(xq, top_k if top_k else self.top_k)
        D = self._convert_cosine_similarity_to_probability(D)
        retrieved_output: RetrieverOutputType = self._to_retriever_output(Ind, D)
        for i, output in enumerate(retrieved_output):
            output.query = queries[i]
        return retrieved_output

    def __call__(
        self, query_or_queries: Union[str, List[str]], top_k: Optional[int] = None
    ) -> RetrieverOutputType:
        response = self.retrieve(query_or_queries=query_or_queries, top_k=top_k)
        return response

    def extra_repr(self) -> str:
        s = f"top_k={self.top_k}, dimensions={self.dimensions}, "
        return s
