"""All data types used by Parameter, Optimizer, AdalComponent, and Trainer."""

from typing import List, Dict, Any, Optional
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime


from adalflow.core import DataClass


class ParameterType(Enum):
    __doc__ = """Enum for the type of parameter to compute the loss with, and to inform the optimizer."""

    PROMPT = (
        "prompt",
        "Instruction to the language model on task, data, and format.",
    )
    DEMOS = ("demos", "A few examples to guide the language model.")
    # INSTANCE = ("instance", "Focus on fixing issues of this specific example.")
    GENERATOR_OUTPUT = (
        "generator_output",
        "The output of the generator.",
    )  # use raw response or error message as data, full response in full_response
    RETRIEVER_OUTPUT = ("retriever_output", "The output of the retriever.")
    LOSS_OUTPUT = ("loss", "The loss value.")
    SUM_OUTPUT = ("sum", "The sum of the losses.")
    GRADIENT = ("gradient", "A gradient parameter.")
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
class EvaluationResult(DataClass):
    """A single evaluation of task pipeline response to a score in range [0, 1]."""

    score: float = field(
        default=0.0, metadata={"desc": "The score of the evaluation in range [0, 1]."}
    )
    feedback: str = field(
        default="",
        metadata={
            "desc": "Feedback on the evaluation, including reasons for the score."
        },
    )


@dataclass
class PromptData:
    id: str  # each parameter's id
    name: str  # each parameter's name
    data: str  # each parameter's data
    requires_opt: bool = field(
        default=True, metadata={"desc": "Whether this parameter requires optimization"}
    )


@dataclass
class TrainerStepResult(DataClass):
    step: int = field(default=0, metadata={"desc": "Step number"})
    val_score: Optional[float] = field(
        default=None,
        metadata={
            "desc": "Validation score. Usually a smaller set than test set to chose the best parameter value."
        },
    )
    test_score: Optional[float] = field(default=None, metadata={"desc": "Test score"})
    attempted_val_score: Optional[float] = field(
        default=None, metadata={"desc": "Attempted validation score"}
    )
    prompt: Optional[List[PromptData]] = field(
        default=None, metadata={"desc": "Optimized prompts for this step"}
    )


@dataclass
class TrainerResult(DataClass):
    steps: List[int] = field(default_factory=list, metadata={"desc": "List of steps"})
    val_scores: List[float] = field(
        default_factory=list, metadata={"desc": "List of validation scores"}
    )
    test_scores: List[float] = field(
        default_factory=list, metadata={"desc": "List of test scores"}
    )
    prompts: List[List[PromptData]] = field(
        default_factory=list, metadata={"desc": "List of optimized prompts"}
    )
    step_results: List[TrainerStepResult] = field(
        default_factory=list,
        metadata={"desc": "List of step results, in an aggregated form"},
    )
    effective_measure: Dict[str, Dict] = field(
        default_factory=dict,
        metadata={"desc": "Effective measures of the constrained training strategy"},
    )
    time_stamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    trainer_state: Dict[str, Any] = field(
        default=None, metadata={"desc": "Save the most detailed state of the trainer"}
    )


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
