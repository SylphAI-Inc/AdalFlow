from .llm_text_loss import LLMAsTextLoss
from .textual_grad_desc import TextualGradientDescent
from .text_loss_with_eval_fn import EvalFnToTextLoss

from lightrag.utils.registry import EntityMapping

__all__ = [
    "LLMAsTextLoss",
    "EvalFnToTextLoss",
    "TextualGradientDescent",
]

for name in __all__:
    EntityMapping.register(name, globals()[name])
