from .bm25_retriever import (
    BM25Retriever,
    split_text_by_word_fn,
    split_text_by_word_fn_then_lower_tokenized,
)
from .llm_retriever import LLMRetriever

from adalflow.utils import LazyImport, OptionalPackages
from adalflow.utils.registry import EntityMapping

FAISSRetriever = LazyImport(
    "adalflow.components.retriever.faiss_retriever.FAISSRetriever",
    OptionalPackages.FAISS,
)

from .reranker_retriever import RerankerRetriever

# from .postgres_retriever import PostgresRetriever

PostgresRetriever = LazyImport(
    "adalflow.components.retriever.postgres_retriever.PostgresRetriever",
    OptionalPackages.SQLALCHEMY,
)

QdrantRetriever = LazyImport(
    "adalflow.components.retriever.qdrant_retriever.QdrantRetriever",
    OptionalPackages.QDRANT,
)

__all__ = [
    "BM25Retriever",
    "LLMRetriever",
    "FAISSRetriever",
    "RerankerRetriever",
    "PostgresRetriever",
    "QdrantRetriever",
    "split_text_by_word_fn",
    "split_text_by_word_fn_then_lower_tokenized",
]


for name in __all__:
    EntityMapping.register(name, globals()[name])
