from enum import Enum, auto


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


# Goal: write arxiv paper on this.

LightRAG_optimizer_notes = [
    "tags like <SYS></SYS> or <SYSTEM></SYSTEM>  or <START_OF_SYSTEM_PROMPT> <END_OF_SYSTEM_PROMPT>can lead to different performance",
]
