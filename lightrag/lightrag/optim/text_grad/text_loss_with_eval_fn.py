"""Adapted from text_grad's String Based Function"""

from typing import Callable, Dict, Union
import logging
from lightrag.optim.text_grad.function import BackwardContext, GradFunction


from lightrag.core import ModelClient
from lightrag.core.component import BaseComponent
from lightrag.optim.parameter import Parameter
from lightrag.eval.base import BaseEvaluator


log = logging.getLogger(__name__)


# TODO: use BaseComponent instead of Component.
class EvalFnToTextLoss(BaseComponent, GradFunction):
    __doc__ = """Convert an evaluation function to a text loss.

    We make it a component for better visualization and serialization.

    Can be used for tasks that have y_gt (ground truth).
    The fn will be fn(y, y_gt) -> metric, and the loss will be a Parameter with
    the evaluation result and can be used to compute gradients."""

    def __init__(
        self,
        eval_fn: Union[Callable, BaseEvaluator],
        eval_fn_desc: str,
        model_client: ModelClient,
        model_kwargs: Dict[str, object],
    ):
        super().__init__()
        self.eval_fn = eval_fn
        self.eval_fn_desc = eval_fn_desc
        # self.loss_llm = Generator(
        #     model_client=model_client,
        #     model_kwargs=model_kwargs,
        #     template=TEXT_LOSS_TEMPLATE,
        # )

    def forward(
        self, kwargs: Dict[str, Parameter], response_desc: str = None
    ) -> Parameter:
        if response_desc is None:
            response_desc = (
                f"Output of EvalFnToTextLoss with eval_fn_desc: {self.eval_fn_desc}"
            )

        response: str = self.eval_fn(**kwargs)

        # Create a parameter
        response: Parameter = Parameter(
            data=response,
            requires_opt=True,
            predecessors=list(kwargs.values()),
            role_desc=response_desc,
        )

        log.info(f"EvalFnToTextLoss: Input: {kwargs}, Output: {response}")
        response.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward,
                response=response,
                eval_fn_desc=self.eval_fn_desc,
                kwargs=kwargs,
            )
        )
        return response

    # def set_backward_engine(self, backward_engine: "Generator"):
    #     if not backward_engine:
    #         raise ValueError("A backward engine is required.")
    #     self.backward_engine = backward_engine
    #     if not self.backward_engine:

    def backward(
        self, response: Parameter, eval_fn_desc: str, kwargs: Dict[str, Parameter]
    ):
        log.info(f"EvalFnToTextLoss: Backward: {response}")
