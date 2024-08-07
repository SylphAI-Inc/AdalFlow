import uuid
from dataclasses import dataclass, field
from lightrag.core.base_data_class import DataClass


@dataclass
class Example(DataClass):
    __doc__ = """A common dataclass for representing examples in a dataset."""
    id: str = field(
        metadata={"desc": "The unique identifier of the example"},
        default=str(uuid.uuid4()),
    )
    question: str = field(
        metadata={"desc": "The question to be answered"}, default=None
    )
    answer: str = field(metadata={"desc": "The answer to the question"}, default=None)


@dataclass
class HotPotQAData(Example):
    __doc__ = """A dataclass for representing examples in the HotPotQA dataset."""
    gold_titles: set = field(
        metadata={"desc": "The set of titles that support the answer"},
        default=None,
    )
