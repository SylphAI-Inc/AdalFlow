from .bm25_retriever import InMemoryBM25Retriever
from .llm_retriever import LLMRetriever

from lightrag.utils import LazyImport, OptionalPackages
from lightrag.utils.registry import EntityMapping

FAISSRetriever = LazyImport(
    "lightrag.components.retriever.faiss_retriever.FAISSRetriever",
    OptionalPackages.FAISS,
)

from .reranker_retriever import RerankerRetriever
from .postgres_retriever import PostgresRetriever

__all__ = [
    "InMemoryBM25Retriever",
    "LLMRetriever",
    "FAISSRetriever",
    "RerankerRetriever",
    "PostgresRetriever",
]


for name in __all__:
    EntityMapping.register(name, globals()[name])
