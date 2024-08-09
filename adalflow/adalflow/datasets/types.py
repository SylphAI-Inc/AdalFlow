import uuid
from dataclasses import dataclass, field
from adalflow.core.base_data_class import DataClass


@dataclass
class BaseData(DataClass):
    __doc__ = """A common dataclass for representing examples in a dataset."""
    id: str = field(
        metadata={"desc": "The unique identifier of the example", "type": "id"},
        default=str(uuid.uuid4()),
    )


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


# TODO: maybe get a way to mark input and output fields, we can implement with exclude automatically
# to generate the to_dict and from_dict methods, when describe the data should exclude the id.
@dataclass
class TrecData(BaseData):
    __doc__ = """A dataclass for representing examples in the TREC dataset."""
    question: str = field(
        metadata={"desc": "The question to be classified", "type": "input"},
        default=None,
    )
    # thought: str = field(
    #     metadata={
    #         "desc": "Your reasoning to classify the question to class_name",
    #     }
    # )
    class_name: str = field(
        metadata={"desc": "The class name", "type": "output"}, default=None
    )

    class_index: int = field(
        metadata={"desc": "The class label, in range [0, 5]", "type": "output"},
        default=None,
    )
