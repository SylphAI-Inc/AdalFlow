from .bm25_retriever import (
    BM25Retriever,
    split_text_by_word_fn,
    split_text_by_word_fn_then_lower_tokenized,
)
from .llm_retriever import LLMRetriever

from lightrag.utils import LazyImport, OptionalPackages
from lightrag.utils.registry import EntityMapping

FAISSRetriever = LazyImport(
    "lightrag.components.retriever.faiss_retriever.FAISSRetriever",
    OptionalPackages.FAISS,
)

from .reranker_retriever import RerankerRetriever

# from .postgres_retriever import PostgresRetriever

PostgresRetriever = LazyImport(
    "lightrag.components.retriever.postgres_retriever.PostgresRetriever",
    OptionalPackages.SQLALCHEMY,
)

__all__ = [
    "BM25Retriever",
    "LLMRetriever",
    "FAISSRetriever",
    "RerankerRetriever",
    "PostgresRetriever",
    "split_text_by_word_fn",
    "split_text_by_word_fn_then_lower_tokenized",
]


for name in __all__:
    EntityMapping.register(name, globals()[name])
