from .function import BackwardContext
from .llm_text_loss import LLMAsTextLoss
from .text_loss_with_eval_fn import EvalFnToTextLoss

from lightrag.utils.registry import EntityMapping

__all__ = ["BackwardContext", "LLMAsTextLoss", "EvalFnToTextLoss"]

for name in __all__:
    EntityMapping.register(name, globals()[name])
