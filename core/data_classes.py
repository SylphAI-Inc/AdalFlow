"""
The data classes used to support core components.
We use dataclass which provides a decorator that automatically adds special methods to classes, such as __init__, __repr__, and __eq__, among others:
"""

from enum import Enum, auto
from typing import List, Dict, Any
from dataclasses import dataclass
import typing
import sys

# if sys.version_info >= (3, 10, 1):
#     Literal = typing.Literal
# else:
#     raise ImportError("Please upgrade to Python 3.10.1 or higher to use Literal")


class ModelType(Enum):
    EMBEDDER = auto()
    LLM = auto()
    UNDEFINED = auto()


@dataclass
class Embedding:
    """
    In sync with api spec, same as openai/types/embedding.py
    """

    embedding: List[float]
    index: int  # match with the index of the input


@dataclass
class Usage:
    """
    In sync with api spec, same as openai/types/create_embedding_response.py
    """

    prompt_tokens: int
    total_tokens: int


@dataclass
class EmbedderResponse:
    data: List[Embedding]
    model: str
    usage: Usage


# class EmbedderOutput:
#     embeddings: List[List[float]]  # batch_size X embedding_size
#     usage: Dict[str, Any]  # api or model usage

#     def __init__(self, embeddings: List[List[float]], usage: Dict[str, Any]):
#         self.embeddings = embeddings
#         self.usage = usage

#     def __repr__(self) -> str:
#         return f"EmbedderOutput(embeddings={self.embeddings[0:5]}, usage={self.usage})"

#     def __str__(self):
#         return self.__repr__()
