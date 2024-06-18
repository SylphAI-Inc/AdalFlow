from .bm25_retriever import *
from .llm_retriever import *

from lightrag.utils.registry import EntityMapping

# FAISS Retriever
try:
    from .faiss_retriever import FAISSRetriever

    EntityMapping.register("FAISSRetriever", FAISSRetriever)
except ImportError as e:
    print(f"Optional module not installed: {e}")

# Reranker
try:
    from .reranker_retriever import *

    EntityMapping.register("RerankerRetriever", RerankerRetriever)
except ImportError as e:
    print(f"Optional module not installed: {e}")
