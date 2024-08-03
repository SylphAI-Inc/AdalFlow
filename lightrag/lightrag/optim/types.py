from typing import List, Dict, Any
from enum import Enum, auto
from dataclasses import dataclass

from lightrag.core import DataClass


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
