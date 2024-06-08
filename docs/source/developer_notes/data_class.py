from dataclasses import dataclass


@dataclass
class TrecData:
    question: str
    label: int


# lightrag

from lightrag.core import DataClass
