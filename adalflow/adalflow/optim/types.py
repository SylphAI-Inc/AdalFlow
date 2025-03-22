"""All data types used by Parameter, Optimizer, AdalComponent, and Trainer."""

from typing import List, Dict, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime


from adalflow.core import DataClass


# TODO: set default optimization
class ParameterType(Enum):
    __doc__ = """Enum for the type of parameter to compute the loss with, and to inform the optimizer.

    The meaning of reach tuple is:
    1. First element: the name of the parameter.
    2. Second element: the description of the parameter.
    3. Third element: whether the parameter is trainable.

    To access each element, use the following:
    1. name: `ParameterType.PROMPT.value`
    2. description: `ParameterType.PROMPT.description`
    3. trainable: `ParameterType.PROMPT.default_trainable`
    """

    # trainable parameters with optimizers
    PROMPT = (
        "prompt",
        "Instruction to the language model on task, data, and format.",
        True,
    )  # optimized by tgd_optimizer
    DEMOS = (
        "demos",
        "A few examples to guide the language model.",
        True,
    )  # optimized by demo_optimizer

    # input and output parameters (similar to tensor, can have grad_opt true, but not trainable)
    INPUT = ("input", "The input to the component.", False)
    OUTPUT = ("output", "The output of the component.", True)
    HYPERPARAM = ("hyperparam", "Hyperparameters/args for the component.", False)

    # the following is a subtype of the output type
    # INSTANCE = ("instance", "Focus on fixing issues of this specific example.")
    GENERATOR_OUTPUT = (
        "generator_output",
        "The output of the generator.",
        True,
    )  # use raw response or error message as data, full response in full_response
    RETRIEVER_OUTPUT = ("retriever_output", "The output of the retriever.", True)
    LOSS_OUTPUT = ("loss", "The loss value.", True)
    SUM_OUTPUT = ("sum", "The sum of the losses.", True)
    NONE = ("none", "", False)

    def __init__(self, value: str, description: str, default_trainable: bool):
        self._value_ = value
        self.description = description
        self.default_trainable = default_trainable

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
    val_score: float = field(
        default=None,
        metadata={
            "desc": "Validation score. Usually a smaller set than test set to chose the best parameter value."
        },
    )
    test_score: float = field(default=None, metadata={"desc": "Test score"})
    attempted_val_score: float = field(
        default=None, metadata={"desc": "Attempted validation score"}
    )
    prompt: List[PromptData] = field(
        default=None, metadata={"desc": "Optimized prompts for this step"}
    )


@dataclass
class TrainerValidateStats:
    """A single evaluation of task pipeline response to a score in range [0, 1]."""

    max_score: float = field(
        default=0.0, metadata={"desc": "The score of the evaluation in range [0, 1]."}
    )
    min_score: float = field(
        default=0.0, metadata={"desc": "The score of the evaluation in range [0, 1]."}
    )
    mean_of_score: float = field(
        default=0.0, metadata={"desc": "The score of the evaluation in range [0, 1]."}
    )
    std_of_score: float = field(
        default=0.0, metadata={"desc": "The score of the evaluation in range [0, 1]."}
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

    step_results: List[TrainerStepResult] = field(
        default_factory=list,
        metadata={"desc": "List of step results, in an aggregated form"},
    )
    effective_measure: Dict[str, Dict] = field(
        default_factory=dict,
        metadata={"desc": "Effective measures of the constrained training strategy"},
    )
    validate_stats: TrainerValidateStats = field(
        default=None,
        metadata={"desc": "Attempted Validation score statistics"},
    )
    time_stamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    total_time: float = field(
        default=0.0, metadata={"desc": "Total time taken for training"}
    )
    test_score: float = field(default=None, metadata={"desc": "Test score"})
    trainer_state: Dict[str, Any] = field(
        default=None, metadata={"desc": "Save the most detailed state of the trainer"}
    )
