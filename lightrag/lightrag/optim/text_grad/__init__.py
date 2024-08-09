from .llm_text_loss import LLMAsTextLoss
from .tgd_optimer import TGDOptimizer
from .text_loss_with_eval_fn import EvalFnToTextLoss

from lightrag.utils.registry import EntityMapping

__all__ = [
    "LLMAsTextLoss",
    "EvalFnToTextLoss",
    "TGDOptimizer",
]

for name in __all__:
    EntityMapping.register(name, globals()[name])
