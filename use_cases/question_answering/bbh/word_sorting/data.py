"""Prepare classes, components, functions that will prepare and post processing the data"""

from dataclasses import dataclass, field
import adalflow as adal

from adalflow.core import DataClass
from adalflow.datasets.big_bench_hard import BigBenchHard
from adalflow.utils.data import subset_dataset


# TODO: user dont have to specify, we can auto generate a dataclass
@dataclass
class QuestionAnswer(DataClass):
    """Dataclass for string output"""

    id: str = field(
        default=None,
        metadata={"desc": "The unique identifier of the example"},
    )

    question: str = field(
        default=None,
        metadata={"desc": "The question to be answered"},
    )

    answer: str = field(
        default=None,
        metadata={"desc": "The raw answer to the question"},  # teacher
    )

    score: float = field(
        default=None,
        metadata={
            "desc": "The score of the answer, in range [0, 1]. The higher the better"
        },
    )  # score can be used as weight for demo, weight = score (the higher the more likely to be sampled)


import re


def _parse_integer_answer(answer: str):
    """A function that parses the last integer from a string using regular expressions."""
    try:
        # Use regular expression to find all sequences of digits
        numbers = re.findall(r"\d+", answer)
        if numbers:
            # Get the last number found
            answer = int(numbers[-1])
        else:
            answer = -1
    except ValueError:
        answer = -1

    return answer


@adal.fun_to_component
def parse_integer_answer(answer: str):
    """A function that parses the last integer from a string using regular expressions."""
    try:
        # Use regular expression to find all sequences of digits
        numbers = re.findall(r"\d+", answer)
        if numbers:
            # Get the last number found
            answer = int(numbers[-1])
        else:
            answer = -1
    except ValueError:
        answer = -1

    return answer


def load_datasets(max_samples: int = None):
    """Load the dataset"""
    train_data = BigBenchHard(split="train", task_name="BBH_word_sorting")
    val_data = BigBenchHard(split="val")
    test_data = BigBenchHard(split="test")

    # Limit the number of samples
    if max_samples:
        train_data = subset_dataset(train_data, max_samples)
        val_data = subset_dataset(val_data, max_samples)
        test_data = subset_dataset(test_data, max_samples)

    return train_data, val_data, test_data


if __name__ == "__main__":
    # TODO: make the int parser to be able to extract the last int

    tests = [
        "Answer: 10",
        "xxxx10XXXX100",
        "Answer: 10.0",
        "Answer: 10.0.0",
        "1\n2 Answer:5",
    ]
    for text in tests:
        print(f"{text} -> {parse_integer_answer(text)}")
