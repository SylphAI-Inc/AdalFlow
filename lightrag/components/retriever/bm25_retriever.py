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
(2) api_client (model)
(3) output_processors

Retriever
(1) 
"""

from typing import List, Dict, Tuple, Optional, Callable, Union
import collections
import heapq
import math
import sys

from multiprocessing import Pool, cpu_count
from functools import partial

from core.tokenizer import Tokenizer
from core.data_classes import RetrieverOutput, RetrieverOutput
from core.retriever import Retriever, RetrieverInputType, RetrieverOutputType


PARAM_K1 = 1.5
PARAM_B = 0.75
IDF_CUTOFF = 0

split_text_by_word_fn = lambda x: x.split()


def split_text_by_token_fn(tokenizer: Tokenizer, x: str) -> List[str]:
    return tokenizer(x)


class InMemoryBM25Retriever(Retriever):
    __doc__ = r"""Fast Implementation of Best Matching 25 ranking function.
    Build index from List[str] where the str is from Document

    IDF(q_i)= log(N/ DF) = log(N/n(q_i) + 0.5)/(n(q_i) + 0.5) + 1, to avoid division by zero and to diminish the weight of terms that occur very frequently in the document set and increase the weight of terms that occur rarely.
    N: total number of documents
    DF/ n(q_i): number of documents containing q_i
    TF/ f(q_i, d): frequency of q_i in document d
    |d|: length of document d in words or tokens
    avgdl: average document length in the corpus (in words or tokens)

    Score(q, d) = sum_{i=1}^{n} IDF(q_i) * (f(q_i, d) * (k1 + 1)) / (f(q_i, d) + k1 * (1 - b + b * |d| / avgdl))

    Attributes
    ----------
    t2d : <token: <doc, freq>>
            Dictionary with terms frequencies for each document in `corpus`.
    idf: <token, idf score>
            Pre computed IDF score for every term.
    doc_len : list of int
            List of document lengths.
    avgdl : float
            Average length of document in `corpus`.
    """

    def __init__(
        self,
        top_k: int = 5,
        # index arguments
        k1: float = PARAM_K1,
        b: float = PARAM_B,
        alpha: float = IDF_CUTOFF,
        split_function: Optional[Callable] = partial(
            split_text_by_token_fn, Tokenizer()
        ),
    ):
        """
        Parameters
        ----------
        corpus : list of list of str
                Given corpus.
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
        alpha: float
                IDF cutoff, terms with a lower idf score than alpha will be dropped. A higher alpha will lower the accuracy
                of BM25 but increase performance
        """
        super().__init__()
        self.k1 = k1
        self.b = b
        self.alpha = alpha
        self.top_k = top_k

        self.split_function = split_function

        self.indexed = False  # this is important to check if the retrieve is possible

    def _apply_split_function(self, documents: List[str]):
        if self.split_function is None:
            raise ValueError("split_function is not defined")
        if not documents:
            print("No documents to split")
            return
        pool = Pool(cpu_count())
        tokenized_documents = pool.map(self.split_function, documents)
        return tokenized_documents

    @property
    def corpus_size(self):
        return len(self.corpus)

    # TODO: better implement this
    def load_index(self, index):
        self.t2d = index["t2d"]
        self.idf = index["idf"]
        self.doc_len = index["doc_len"]
        self.avgdl = index["avgdl"]
        self.k1 = index["k1"]
        self.b = index["b"]
        self.alpha = index["alpha"]

    def save_index(self):
        return {
            "t2d": self.t2d,
            "idf": self.idf,
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
            "k1": self.k1,
            "b": self.b,
            "alpha": self.alpha,
        }

    def reset_index(self):
        self.t2d = {}
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.k1 = PARAM_K1
        self.b = PARAM_B
        self.alpha = IDF_CUTOFF
        self.indexed = False

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
        # start to calculate the DF,TF, IDF
        self.avgdl = 0  # average document length
        self.t2d: Dict[str, Tuple[int, int]] = (
            {}
        )  # term to document, <term: <doc_index, freq>>
        self.idf = {}
        self.doc_len = []
        corpus_size = len(list_of_documents_str)
        for i, document in enumerate(self.tokenized_documents):
            self.doc_len.append(len(document))
            for token in document:
                if token not in self.t2d:
                    self.t2d[token] = {}
                if i not in self.t2d[token]:
                    self.t2d[token][i] = 0
                self.t2d[token][i] += 1
        self.avgdl = sum(self.doc_len) / len(list_of_documents_str)
        to_delete = []
        for token, docs in self.t2d.items():
            idf = math.log(corpus_size - len(docs) + 0.5) - math.log(len(docs) + 0.5)
            self.idf[token] = idf
            # if idf > self.alpha:
            #     self.idf[token] = idf
            # else:
            #     to_delete.append(token)
        print(f"idf: {self.idf}")
        for token in to_delete:
            del self.t2d[token]
        self.average_idf = sum(self.idf.values()) / len(self.idf)
        if self.average_idf < 0:
            print(
                f"Average inverse document frequency is less than zero. Your corpus of {corpus_size} documents"
                " is either too small or it does not originate from natural text. BM25 may produce"
                " unintuitive results.",
                file=sys.stderr,
            )
        self.indexed = True

    # TODO: retriever should output the list of indexes of the documents.
    def retrieve(
        self, query_or_queries: RetrieverInputType, top_k: Optional[int] = None
    ) -> RetrieverOutputType:
        """
        Retrieve the top n documents for the query and return only the indexes of the documents.

        Parameters
        ----------
        query: list of str
                The tokenized query
        documents: list
                The documents to return from
        n: int
                The number of documents to return

        Returns
        -------
        list
                The top n documents
        """

        scores = collections.defaultdict(float)
        if top_k is None:
            top_k = self.top_k

        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        # process queries into lower case and split into tokens
        processed_queries = [query.lower() for query in queries]
        tokenized_queries = [self.split_function(query) for query in processed_queries]
        output = []
        for idx, query in enumerate(tokenized_queries):
            query_response: RetrieverOutput = None
            for token in query:
                if token in self.t2d:
                    for index, freq in self.t2d[token].items():
                        denom_cst = self.k1 * (
                            1 - self.b + self.b * self.doc_len[index] / self.avgdl
                        )
                        scores[index] += (
                            self.idf[token] * freq * (self.k1 + 1) / (freq + denom_cst)
                        )

            retrieved_documents_indexes = [
                i for i in heapq.nlargest(top_k, scores.keys(), key=scores.__getitem__)
            ]
            retrieved_documents_scores = [
                scores[i]
                for i in heapq.nlargest(top_k, scores.keys(), key=scores.__getitem__)
            ]
            query_response = RetrieverOutput(
                doc_indexes=retrieved_documents_indexes,
                query=queries[idx],
                doc_scores=retrieved_documents_scores,
            )
            output.append(query_response)

        print(f"output: {output}")

        return output

    # TODO: enforce typing along the whole lifecycle of the retriever
    def __call__(
        self, query_or_queries: Union[str, List[str]], top_k: Optional[int] = None
    ) -> RetrieverOutputType:
        response = self.retrieve(query_or_queries=query_or_queries, top_k=top_k)
        # if self.output_processors:
        #     response = self.output_processors(response)
        return response

    # def save(self, filename):
    #     with open(f"{filename}.pkl", "wb") as fsave:
    #         pickle.dump(self, fsave, protocol=pickle.HIGHEST_PROTOCOL)

    # @staticmethod
    # def load(filename):
    #     with open(f"{filename}.pkl", "rb") as fsave:
    #         return pickle.load(fsave)
