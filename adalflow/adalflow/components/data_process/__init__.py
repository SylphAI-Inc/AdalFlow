"""Components here are used for data processing/transformation."""

from .text_splitter import TextSplitter
from .data_components import RetrieverOutputToContextStr
from adalflow.utils.registry import EntityMapping


__all__ = ["TextSplitter", "RetrieverOutputToContextStr"]

for name in __all__:
    EntityMapping.register(name, globals()[name])
