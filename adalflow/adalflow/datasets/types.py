import uuid
from dataclasses import dataclass, field
from typing import Dict
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
class GSM8KData(Example):
    __doc__ = """A dataclass for representing examples in the GSM8K dataset.

    You can reset the output fields:

    .. code-block:: python

        GSM8KData.set_output_fields(["answer"])
    """
    gold_reasoning: str = field(
        metadata={"desc": "The ground truth reasoning for the answer"}, default=None
    )
    reasoning: str = field(
        metadata={"desc": "The reasoning for the answer"}, default=None
    )  # your model's reasoning

    __input_fields__ = ["question"]
    __output_fields__ = ["reasoning", "answer"]  # default output fields


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

    __input_fields__ = ["question"]
    __output_fields__ = ["answer"]

    # @staticmethod
    # def from_dict(d: Dict[str, Any]) -> "HotPotQAData":
    #     # Preprocess gold_titles
    #     if "gold_titles" in d and isinstance(d["gold_titles"], str):
    #         try:
    #             d["gold_titles"] = json.loads(d["gold_titles"])
    #         except json.JSONDecodeError:
    #             # Replace single quotes with double quotes
    #             fixed_str = d["gold_titles"].replace("'", '"')
    #             d["gold_titles"] = set(json.loads(fixed_str))

    #     # Preprocess context
    #     if "context" in d and isinstance(d["context"], str):
    #         try:
    #             d["context"] = json.loads(d["context"])
    #         except json.JSONDecodeError:
    #             fixed_str = d["context"].replace("'", '"')
    #             d["context"] = json.loads(fixed_str)

    #     return HotPotQAData(**d)


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


if __name__ == "__main__":
    # test the hotpotqa data
    data = HotPotQAData(
        question="What is the capital of France?",
        answer="Paris",
        gold_titles=set(["Paris", "France"]),
        context={"Paris": "The capital of France"},
    )

    data_dict = data.to_dict()
    print("data_dict", data_dict)
    data = HotPotQAData.from_dict(data_dict)
    print("data", data)

    from adalflow.utils.file_io import save_json, load_json

    # save json
    save_json(data_dict, f="task.json")
    # load json
    data_dict_loaded = load_json(f="task.json")

    print("data_dict_loaded", data_dict_loaded)

    # restore the data
    data_restored = HotPotQAData.from_dict(data_dict_loaded)
    print("data_restored", data_restored)
