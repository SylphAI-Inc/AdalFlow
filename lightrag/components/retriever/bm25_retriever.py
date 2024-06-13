r""" Implementation based on : https://en.wikipedia.org/wiki/Okapi_BM25
TODO: Trotmam et al, Improvements to BM25 and Language Models Examined
Retrieval is highly dependent on the database.

db-> transformer -> (index) should be a pair
LocalDocumentDB:  [Local Document RAG]
(1) algorithm, (2) index, build_index_from_documents (3) retrieve (top_k, query)

What algorithm will do for LocalDocumentDB:
(1) Build_index_from_documents (2) retrieval initialization (3) retrieve (top_k, query), potentially with score.

InMemoryRetriever: (Component)
(1) load_documents (2) build_index_from_documents (3) retrieve (top_k, query)

PostgresDB:
(1) sql_query for retrieval (2) pg_vector for retrieval (3) retrieve (top_k, query)

MemoryDB:
(1) chat_history (2) provide different retrieval methods, allow specify retrievel method at init.

Generator:
(1) prompt
(2) model_client (model)
(3) output_processors

Retriever
(1) 
"""

from typing import List, Dict, Optional, Callable, Any, Union
import numpy as np
import heapq
import math
import logging

from multiprocessing import Pool, cpu_count
from functools import partial

from lightrag.core.tokenizer import Tokenizer
from lightrag.core.types import RetrieverOutput, RetrieverOutput
from lightrag.core.retriever import Retriever, RetrieverInputType, RetrieverOutputType

log = logging.getLogger(__name__)

PARAM_K1 = 1.5
PARAM_B = 0.75
PARAM_EPSILON = 0.25


def split_text_by_word_fn(x: str) -> List[str]:
    x = x.lower()
    return x.split(" ")


def split_text_by_token_fn(tokenizer: Tokenizer, x: str) -> List[str]:
    x = x.lower()
    return tokenizer(x)


# TODO: explain epsilon
class InMemoryBM25Retriever(Retriever):
    __doc__ = r"""Fast Implementation of Best Matching 25 ranking function.

    .. math::

        \text{idf}(q_i) = \epsilon \log\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}\right)

        \text{score}(q, d) = \sum_{i=1}^{n} \text{idf}(q_i) \cdot \frac{f(q_i, d) \cdot (k1 + 1)}{f(q_i, d) + k1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}

    Explanation:
        - IDF(q_i) is the inverse document frequency of term q_i, which measures how important the term is. To avoid division by zero, 0.5 is added to the denominator, also for diminishing the weight of terms that occur very frequently in the document set and increase the weight of terms that occur rarely.
        - f(q_i, d) is the term frequency of term q_i in document d, which measures how often the term occurs in the document. The term frequency is normalized by dividing the raw term frequency by the document length.
        - |d| is the length of the document d in words or tokens.
        - avgdl is the average document length in the corpus.
        - N is the total number of documents in the corpus.
        - n(q_i) is the number of documents containing term q_i.

    References:
        [1] https://en.wikipedia.org/wiki/Okapi_BM25

        [2] https://github.com/dorianbrown/rank_bm25

    Args:
        top_k : (int): The number of documents to return
        k1 : float
                Constant used for influencing the term frequency saturation. After saturation is reached, additional
                presence for the term adds a significantly less additional score. According to [1]_, experiments suggest
                that 1.2 < k1 < 2 yields reasonably good results, although the optimal value depends on factors such as
                the type of documents or queries.
        b : float
                Constant used for influencing the effects of different document lengths relative to average document length.
                When b is bigger, lengthier documents (compared to average) have more impact on its effect. According to
                [1]_, experiments suggest that 0.5 < b < 0.8 yields reasonably good results, although the optimal value
                depends on factors such as the type of documents or queries.

        epsilon: float
                A small 
               


    Examples:


    """

    def __init__(
        self,
        top_k: int = 5,
        # index arguments
        k1: float = PARAM_K1,
        b: float = PARAM_B,
        epsilon: float = PARAM_EPSILON,
        split_function: Optional[Callable] = partial(
            split_text_by_token_fn, Tokenizer()
        ),
    ):
        r"""
        - nd: <token, freq> (n(q_i) in the formula)
        - t2d: <token, <doc_index, freq>> (f(q_i, d) in the formula)
        - idf: <token, idf> (idf(q_i) in the formula)
        - doc_len: list of document lengths (|d| in the formula)
        - avgdl: average document length in the corpus (avgdl in the formula)
        - corpus_size: total number of documents in the corpus (N in the formula)
        """
        super().__init__()
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.top_k = top_k
        self.split_function = split_function
        self.indexed = False  # this is important to check if the retrieve is possible
        self.index_keys = ["nd", "t2d", "idf", "doc_len", "avgdl", "corpus_size"]

    def _apply_split_function(self, documents: List[str]):
        if self.split_function is None:
            raise ValueError("split_function is not defined")
        if not documents:
            log.warning("No documents to split")
            return
        tokenized_documents = [self.split_function(doc) for doc in documents]
        # pool = Pool(cpu_count())
        # tokenized_documents = pool.map(self.split_function, documents)
        return tokenized_documents

    def load_index(self, index: Dict[str, Any]):

        self.indexed = True
        if not all(key in index for key in self.index_keys):
            raise ValueError(
                f"Index keys are not complete. Expected keys: {self.index_keys}"
            )
        self.indexed = True
        for key, value in index.items():
            setattr(self, key, value)

    def get_index(self) -> Dict[str, Any]:
        if not self.indexed:
            raise ValueError(
                "Index is not built or loaded. Please either build the index or load it first."
            )
        return {key: getattr(self, key) for key in self.index_keys}

    def reset_index(self):
        self.t2d = []
        self.nd = {}
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.indexed = False

    def _initialize(self, corpus: List[List[str]]):
        r"""Initialize the term to document dictionary with the term frequencies in each document"""
        self.t2d: List[Dict[str, int]] = []  # term freuqency in each document
        self.nd: Dict[str, int] = {}  # number of documents containing the term
        self.avgdl = 0  # average document length
        self.doc_len = []  # list of document lengths
        self.corpus_size = len(corpus)
        for document in corpus:
            self.doc_len.append(len(document))
            term_freq = {}
            for token in document:
                if token not in term_freq:
                    term_freq[token] = 0
                term_freq[token] += 1

            self.t2d.append(term_freq)

            for word, _ in term_freq.items():
                if word not in self.nd:
                    self.nd[word] = 0
                self.nd[word] += 1

        self.avgdl = sum(self.doc_len) / len(corpus)

    def _calc_idf(self):
        idf_sum = 0
        negative_idf = (
            []
        )  # idf can be negative if word is too common: more than half of the documents
        self.idf: Dict[str, float] = {}
        for token, freq in self.nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[token] = idf
            idf_sum += idf
            if idf < 0:
                negative_idf.append(token)
        self.average_idf = idf_sum / len(self.nd)  # average idf for each term

        eps = self.epsilon * self.average_idf
        for token in negative_idf:
            self.idf[token] = eps

    def _get_scores(self, query: List[str]) -> List[float]:
        r"""Calculate the BM25 score for the query and the documents in the corpus

        Args:
            query: List[str]: The tokenized query
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.t2d])
            score += (self.idf.get(q) or 0) * (
                q_freq
                * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score.tolist()

    def _get_batch_scores(self, query: List[str], doc_ids: List[int]) -> List[float]:
        r"""Calculate the BM25 score for the query and the documents in the corpus

        Args:
            query: List[str]: The tokenized query
            doc_ids: List[int]: The list of document indexes to calculate the score
        """
        assert all(di < len(self.t2d) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.t2d[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (
                q_freq
                * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score.tolist()

    def build_index_from_documents(
        self,
        documents: List[str],
        # input_field_map_func: Callable[[Any], str] = lambda x: x.text,
    ):
        r"""Built index from the `text` field of each document in the list of documents"""

        # make a copy of the documents
        # list_of_documents_str = [input_field_map_func(doc) for doc in documents]
        list_of_documents_str = documents.copy()
        self.tokenized_documents = self._apply_split_function(list_of_documents_str)
        self._initialize(self.tokenized_documents)
        self._calc_idf()
        self.indexed = True

    def retrieve(
        self, query_or_queries: RetrieverInputType, top_k: Optional[int] = None
    ) -> RetrieverOutputType:
        """
        Retrieve the top n documents for the query and return only the indexes of the documents.

        Args:
            query_or_queries: Union[str, List[str]]: The query or list of queries
            top_k: Optional[int]: The number of documents to return
        """
        if not self.indexed:
            raise ValueError("Index is not built. Please build the index first.")

        top_k = top_k or self.top_k
        output: RetrieverOutputType = []
        if isinstance(query_or_queries, str):
            query_or_queries = [query_or_queries]
        elif isinstance(query_or_queries, list):
            pass
        else:
            raise ValueError("query_or_queries should be a string or a list of strings")
        # process each query
        for query in query_or_queries:
            tokens = self.split_function(query)
            scores = self._get_scores(tokens)
            top_k_idx = heapq.nlargest(top_k, range(len(scores)), scores.__getitem__)
            top_k_scores = [scores[i] for i in top_k_idx]
            output.append(
                RetrieverOutput(doc_indices=top_k_idx, doc_scores=top_k_scores)
            )
        return output

    def __call__(
        self, query_or_queries: Union[str, List[str]], top_k: Optional[int] = None
    ) -> RetrieverOutputType:
        response = self.retrieve(query_or_queries=query_or_queries, top_k=top_k)
        # if self.output_processors:
        #     response = self.output_processors(response)
        return response
