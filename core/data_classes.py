"""
The data classes used to support core components
"""

from enum import Enum, auto


class ModelType(Enum):
    EMBEDDER = auto()
    LLM = auto()
    UNDEFINED = auto()
