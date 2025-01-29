"""Semantic search/embedding-based retriever using FAISS."""

from typing import (
    List,
    Optional,
    Sequence,
    Union,
    Dict,
    overload,
    Literal,
    Any,
    Callable,
)
import numpy as np
import logging
import os


from adalflow.core.retriever import Retriever
from adalflow.core.embedder import Embedder
from adalflow.core.types import (
    RetrieverOutput,
    RetrieverOutputType,
    RetrieverStrQueryType,
    EmbedderOutputType,
)
from adalflow.core.functional import normalize_np_array, is_normalized

from adalflow.utils.lazy_import import safe_import, OptionalPackages

safe_import(OptionalPackages.FAISS.value[0], OptionalPackages.FAISS.value[1])
import faiss

log = logging.getLogger(__name__)

FAISSRetrieverDocumentEmbeddingType = Union[List[float], np.ndarray]  # single embedding
FAISSRetrieverDocumentsType = Sequence[FAISSRetrieverDocumentEmbeddingType]

FAISSRetrieverEmbeddingQueryType = Union[
    List[float], List[List[float]], np.ndarray
]  # single embedding or list of embeddings
FAISSRetrieverQueryType = Union[RetrieverStrQueryType, FAISSRetrieverEmbeddingQueryType]
FAISSRetrieverQueriesType = Sequence[FAISSRetrieverQueryType]
FAISSRetrieverQueriesStrType = Sequence[RetrieverStrQueryType]
FAISSRetrieverQueriesEmbeddingType = Sequence[FAISSRetrieverEmbeddingQueryType]

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class FAISSRetriever(
    Retriever[FAISSRetrieverDocumentEmbeddingType, FAISSRetrieverQueryType]
):
    __doc__ = r"""Semantic search/embedding-based retriever using FAISS.

    To use the retriever, you can either pass the index embeddings from the :meth:`__init__` or use the :meth:`build_index_from_documents` method.


    Args:
        embedder (Embedder, optimal): The embedder component to use for converting the queries in string format to embeddings.
            Ensure the vectorizer is exactly the same as the one used to the embeddings in the index.
        top_k (int, optional): Number of chunks to retrieve. Defaults to 5.
        dimensions (Optional[int], optional): Dimension of the embeddings. Defaults to None. It can automatically infer the dimensions from the first chunk.
        documents (Optional[FAISSRetrieverDocumentType], optional): List of embeddings. Format can be List[List[float]] or List[np.ndarray]. Defaults to None.
        metric (Literal["cosine", "euclidean", "prob"], optional): The metric to use for the retrieval. Defaults to "prob" which converts cosine similarity to probability.

    How FAISS works:

    The retriever uses in-memory Faiss index to retrieve the top k chunks
    d: dimension of the vectors
    xb: number of vectors to put in the index
    xq: number of queries
    The data type dtype must be float32.

    Note: When the num of chunks are less than top_k, the last columns will be -1

    Other index options:
    - faiss.IndexFlatL2: L2 or Euclidean distance, [-inf, inf]
    - faiss.IndexFlatIP: Inner product of embeddings (inner product of normalized vectors will be cosine similarity, [-1, 1])

    We choose cosine similarity and convert it to range [0, 1] by adding 1 and dividing by 2 to simulate probability in [0, 1]

    Install FAISS:

    As FAISS is optional package, you can install it with pip for cpu version:
    ```bash
    pip install faiss-cpu
    ```
    For GPU version:
    You might have to use conda to install faiss-gpu:https://github.com/facebookresearch/faiss/wiki/Installing-Faiss

    References:
    - FAISS: https://github.com/facebookresearch/faiss
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        top_k: int = 5,
        dimensions: Optional[int] = None,
        documents: Optional[Any] = None,
        document_map_func: Optional[
            Callable[[Any], FAISSRetrieverDocumentEmbeddingType]
        ] = None,
        metric: Literal["cosine", "euclidean", "prob"] = "prob",
    ):
        super().__init__()

        self.reset_index()

        self.dimensions = dimensions
        self.embedder = embedder  # used to vectorize the queries
        self.top_k = top_k
        self.metric = metric
        if self.metric == "cosine" or self.metric == "prob":
            self._faiss_index_type = faiss.IndexFlatIP
            self._needs_normalized_embeddings = True
        elif self.metric == "euclidean":
            self._faiss_index_type = faiss.IndexFlatL2
            self._needs_normalized_embeddings = False
        else:
            raise ValueError(f"Invalid metric: {self.metric}")

        if documents:
            self.documents = documents
            self.build_index_from_documents(documents, document_map_func)

    def reset_index(self):
        self.index = None
        self.total_documents: int = 0
        self.documents: Sequence[Any] = None
        self.xb: np.ndarray = None
        self.dimensions: Optional[int] = None
        self.indexed: bool = False

    def _preprare_faiss_index_from_np_array(self, xb: np.ndarray):
        r"""Prepare the faiss index from the numpy array."""
        if not self.dimensions:
            self.dimensions = self.xb.shape[1]
        else:
            assert (
                self.dimensions == self.xb.shape[1]
            ), f"Dimension mismatch: {self.dimensions} != {self.xb.shape[1]}"
        self.total_documents = xb.shape[0]

        self.index = self._faiss_index_type(self.dimensions)
        self.index.add(xb)
        self.indexed = True

    def build_index_from_documents(
        self,
        documents: Sequence[Any],
        document_map_func: Optional[
            Callable[[Any], FAISSRetrieverDocumentEmbeddingType]
        ] = None,
    ):
        r"""Build index from embeddings.

        Args:
            documents: List of embeddings. Format can be List[List[float]] or List[np.ndarray]

        If you are using Document format, pass them as [doc.vector for doc in documents]
        """
        if document_map_func:
            assert callable(document_map_func), "document_map_func should be callable"
            documents = [document_map_func(doc) for doc in documents]
        try:
            self.documents = documents

            # convert to numpy array
            if not isinstance(documents, np.ndarray) and isinstance(
                documents[0], Sequence
            ):
                # ensure all the embeddings are of the same size
                assert all(
                    len(doc) == len(documents[0]) for doc in documents
                ), "All embeddings should be of the same size"
                self.xb = np.array(documents, dtype=np.float32)
            else:
                self.xb = documents
            if self._needs_normalized_embeddings:
                first_vector = self.xb[0]
                if not is_normalized(first_vector):
                    log.warning(
                        "Embeddings are not normalized, normalizing the embeddings"
                    )
                    self.xb = normalize_np_array(self.xb)

            self._preprare_faiss_index_from_np_array(self.xb)
            log.info(f"Index built with {self.total_documents} chunks")
        except Exception as e:
            log.error(f"Error building index: {e}, resetting the index")
            # reset the index
            self.reset_index()
            raise e

    def _convert_cosine_similarity_to_probability(self, D: np.ndarray) -> np.ndarray:
        D = (D + 1) / 2
        D = np.round(D, 3)
        return D

    def _to_retriever_output(
        self, Ind: np.ndarray, D: np.ndarray
    ) -> RetrieverOutputType:
        r"""Convert the indices and distances to RetrieverOutputType format."""
        output: RetrieverOutputType = []
        # Step 1: Filter out the -1, -1 columns along with its scores when top_k > len(chunks)
        if -1 in Ind:
            valid_columns = ~np.any(Ind == -1, axis=0)

            D = D[:, valid_columns]
            Ind = Ind[:, valid_columns]
        # Step 2: processing rows (one query at a time)
        for row in zip(Ind, D):
            indices, distances = row
            # convert from numpy to list
            retrieved_documents_indices = indices.tolist()
            retrieved_documents_scores = distances.tolist()
            output.append(
                RetrieverOutput(
                    doc_indices=retrieved_documents_indices,
                    doc_scores=retrieved_documents_scores,
                )
            )

        return output

    def retrieve_embedding_queries(
        self,
        input: FAISSRetrieverQueriesEmbeddingType,
        top_k: Optional[int] = None,
    ) -> RetrieverOutputType:
        if not self.indexed or self.index.ntotal == 0:
            raise ValueError(
                "Index is empty. Please set the chunks to build the index from"
            )
        # check if the input is List, convert to numpy array
        try:
            if not isinstance(input, np.ndarray):
                xq = np.array(input, dtype=np.float32)
            else:
                xq = input
        except Exception as e:
            log.error(f"Error converting input to numpy array: {e}")
            raise e

        D, Ind = self.index.search(xq, top_k if top_k else self.top_k)
        if self.metric == "prob":
            D = self._convert_cosine_similarity_to_probability(D)
        output: RetrieverOutputType = self._to_retriever_output(Ind, D)
        return output

    def retrieve_string_queries(
        self,
        input: Union[str, List[str]],
        top_k: Optional[int] = None,
    ) -> RetrieverOutputType:
        r"""Retrieve the top k chunks given the query or queries in string format.

        Args:
            input: The query or list of queries in string format. Note: ensure the maximum number of queries fits into the embedder.
            top_k: The number of chunks to retrieve. When top_k is not provided, it will use the default top_k set during initialization.

        When top_k is not provided, it will use the default top_k set during initialization.
        """
        if not self.indexed or self.index.ntotal == 0:
            raise ValueError(
                "Index is empty. Please set the chunks to build the index from"
            )
        queries = [input] if isinstance(input, str) else input
        # filter out empty queries
        valid_queries: List[str] = []
        record_map: Dict[int, int] = (
            {}
        )  # final index : the position in the initial queries
        for i, q in enumerate(queries):
            if not q:
                log.warning("Empty query found, skipping")
                continue
            valid_queries.append(q)
            record_map[len(valid_queries) - 1] = i
        # embed the queries, assume the length fits into a batch.
        try:
            embeddings: EmbedderOutputType = self.embedder(valid_queries)
            queries_embeddings: List[List[float]] = [
                data.embedding for data in embeddings.data
            ]

        except Exception as e:
            log.error(f"Error embedding queries: {e}")
            raise e
        xq = np.array(queries_embeddings, dtype=np.float32)
        D, Ind = self.index.search(xq, top_k if top_k else self.top_k)
        D = self._convert_cosine_similarity_to_probability(D)

        output: RetrieverOutputType = [
            RetrieverOutput(doc_indices=[], query=query) for query in queries
        ]
        retrieved_output: RetrieverOutputType = self._to_retriever_output(Ind, D)

        # fill in the doc_indices and score for valid queries
        for i, per_query_output in enumerate(retrieved_output):
            initial_index = record_map[i]
            output[initial_index].doc_indices = per_query_output.doc_indices
            output[initial_index].doc_scores = per_query_output.doc_scores

        return output

    @overload
    def call(
        self,
        input: FAISSRetrieverQueriesEmbeddingType,
        top_k: Optional[int] = None,
    ) -> RetrieverOutputType:
        r"""Retrieve the top k chunks given the query or queries in embedding format."""
        ...

    @overload
    def call(
        self,
        input: FAISSRetrieverQueriesStrType,
        top_k: Optional[int] = None,
    ) -> RetrieverOutputType:
        r"""Retrieve the top k chunks given the query or queries in string format."""
        ...

    def call(
        self,
        input: FAISSRetrieverQueriesType,
        top_k: Optional[int] = None,
    ) -> RetrieverOutputType:
        r"""Retrieve the top k chunks given the query or queries in embedding or string format."""
        assert (
            self.indexed
        ), "Index is not built. Please build the index using build_index_from_documents"
        if isinstance(input, str) or (
            isinstance(input, Sequence) and isinstance(input[0], str)
        ):
            assert self.embedder, "Embedder is not provided"
            return self.retrieve_string_queries(input, top_k)
        else:
            return self.retrieve_embedding_queries(input, top_k)

    def _extra_repr(self) -> str:
        s = f"top_k={self.top_k}"
        if self.metric:
            s += f", metric={self.metric}"
        if self.dimensions:
            s += f", dimensions={self.dimensions}"
        if self.documents:
            s += f", total_documents={self.total_documents}"
        return s
