"""Components here are used for data processing/transformation."""

from .document_splitter import DocumentSplitter
from .data_components import ToEmbeddings, RetrieverOutputToContextStr
from lightrag.utils.registry import EntityMapping


__all__ = ["DocumentSplitter", "ToEmbeddings", "RetrieverOutputToContextStr"]

for name in __all__:
    EntityMapping.register(name, globals()[name])
