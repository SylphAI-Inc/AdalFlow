from .llm_text_loss import LLMAsTextLoss
from .tgd_optimizer import TGDOptimizer
from .text_loss_with_eval_fn import EvalFnToTextLoss
from .ops import sum_ops, Sum

from adalflow.utils.registry import EntityMapping

__all__ = [
    "LLMAsTextLoss",
    "EvalFnToTextLoss",
    "TGDOptimizer",
    "sum_ops",
    "Sum",
]

for name in __all__:
    EntityMapping.register(name, globals()[name])
