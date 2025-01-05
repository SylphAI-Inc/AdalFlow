import uuid
from dataclasses import dataclass, field
from typing import Dict
from adalflow.core.base_data_class import DataClass
import json
from typing import Any


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
    context: Dict[str, object] = field(
        metadata={"desc": "The context of the question"},
        default=None,
    )

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "HotPotQAData":
        # Preprocess gold_titles
        if "gold_titles" in d and isinstance(d["gold_titles"], str):
            try:
                d["gold_titles"] = json.loads(d["gold_titles"])
            except json.JSONDecodeError:
                # Replace single quotes with double quotes
                fixed_str = d["gold_titles"].replace("'", '"')
                d["gold_titles"] = set(json.loads(fixed_str))

        # Preprocess context
        if "context" in d and isinstance(d["context"], str):
            try:
                d["context"] = json.loads(d["context"])
            except json.JSONDecodeError:
                fixed_str = d["context"].replace("'", '"')
                d["context"] = json.loads(fixed_str)

        return HotPotQAData(**d)


@dataclass
class TrecData(BaseData):
    __doc__ = """A dataclass for representing examples in the TREC dataset."""
    question: str = field(
        metadata={"desc": "The question to be classified"},
        default=None,
    )
    class_name: str = field(
        metadata={"desc": "One of {ABBR, ENTY, DESC, HUM, LOC, NUM}"},
        default=None,
    )
    class_index: int = field(
        metadata={"desc": "The class label, in range [0, 5]"},
        default=-1,
    )

    __input_fields__ = ["question"]  # follow this order too.
    __output_fields__ = ["class_name", "class_index"]
