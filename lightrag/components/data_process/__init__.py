"""Components here are used for data processing/transformation."""

from .text_splitter import TextSplitter
from .data_components import ToEmbeddings, RetrieverOutputToContextStr
from lightrag.utils.registry import EntityMapping


__all__ = ["TextSplitter", "ToEmbeddings", "RetrieverOutputToContextStr"]

for name in __all__:
    EntityMapping.register(name, globals()[name])
