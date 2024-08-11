from typing import List, Dict, Any
from enum import Enum, auto
from dataclasses import dataclass

from adalflow.core import DataClass


class ParameterType(Enum):
    """Enum for the type of parameter to compute the loss with, and to inform the optimizer."""

    PROMPT = (
        "prompt",
        "Need to be generic and you can not modify it based on a single example.",
    )
    DEMOS = ("demos", "A few examples to guide the language model.")
    # INSTANCE = ("instance", "Focus on fixing issues of this specific example.")
    GENERATOR_OUTPUT = (
        "generator_output",
        "The output of the generator.",
    )  # use raw response or error message as data, full response in full_response
    RETRIEVER_OUTPUT = ("retriever_output", "The output of the retriever.")
    NONE = ("none", "")

    def __init__(self, value, description):
        self._value_ = value
        self.description = description

    def __str__(self):
        """Return a string representation that includes the enum's value and description."""
        return f"{self.value} ({self.description})"

    def __repr__(self):
        """Return an unambiguous representation that is useful for debugging."""
        return f"<{self.__class__.__name__}.{self.name}: {self.value}, '{self.description}'>"


@dataclass
class PromptData:
    id: str  # each parameter's id
    alias: str  # each parameter's alias
    data: str  # each parameter's data


@dataclass
class TrainerResult(DataClass):
    steps: List[int]
    val_scores: List[float]
    test_scores: List[float]
    prompts: List[List[PromptData]]
    trainer_state: Dict[str, Any] = None
    effective_measure: Dict[str, Dict] = None  # stage


@dataclass
class FewShotConfig:
    raw_shots: int  # raw shots
    bootstrap_shots: int  # bootstrap shots


class OptimizeGoal(Enum):
    # 1. Similar to normal model auto-grad
    LLM_SYS_INSTRUCTION = auto()  # fixed system prompt instruction across all calls
    LLM_PROMP_TEMPLATE = (
        auto()
    )  # fixed prompt template , the tags and format can have a big impact on the performance
    LLM_SYS_EXAMPLE = (
        auto()
    )  # few-shot examples (fixed across all calls) => vs dynamic examples
    DYNAMIC_FEW_SHOT_EXAMPLES = auto()  # dynamic examples leverage retrieval
    LLM_RESPONSE = (
        auto()
    )  # similar to reflection, to optimize the response with optimizer
    HYPERPARAMETER_TUNING = auto()  # optimize hyperparameters


# Goal: The optimization method can be potentially used for hyperparameter tuning too

LightRAG_optimizer_notes = [
    "tags like <SYS></SYS> or <SYSTEM></SYSTEM>  or <START_OF_SYSTEM_PROMPT> <END_OF_SYSTEM_PROMPT>can lead to different performance",
    "System prompt",
    "output format, the description of field",
]
