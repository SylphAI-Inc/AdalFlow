"""Adapted from text_grad's String Based Function"""

from typing import Callable, Dict, Union
import logging
from lightrag.optim.text_grad.function import BackwardContext, GradFunction


from lightrag.core import ModelClient
from lightrag.core.generator import BackwardEngine
from lightrag.core.component import BaseComponent
from lightrag.optim.parameter import Parameter
from lightrag.optim.text_grad.backend_engine_prompt import EVALUATE_VARIABLE_INSTRUCTION
from lightrag.core.prompt_builder import Prompt
from lightrag.eval.base import BaseEvaluator


log = logging.getLogger(__name__)

###  First part of the user prompt in the backward engine:  Eval function context
CONVERSATION_TEMPLATE_STRING = r"""Eval Function Description: {{eval_fn_desc}}
<INPUTS_TO_FUNCTION> {{input_str}} </INPUTS_TO_FUNCTION>
<OUTPUT_OF_FUNCTION> {{response_value}} </OUTPUT_OF_FUNCTION>"""


# Does not have gradient on the output, the loss function of the backpropagation chain
CONVERSATION_START_INSTRUCTION_STRING_FN_BASE = r"""You will give feedback to a variable with the following role:
<ROLE> {{variable_desc}} </ROLE>.
Here is an evaluation of the variable using the eval function:
{{conversation}}"""

# Has the gradient on the output, the layer in the backpropagation chain
# Conversation will be provided differently.
CONVERSATION_START_INSTRUCTION_STRING_FN_CHAIN = r"""You will give feedback to a variable with the following role:
<ROLE> {{variable_desc}} </ROLE>.
Here is the evaluation of the eval function with inputs and outputs:
{{conversation}}"""

# Third part of the user prompt
OBJECTIVE_INSTRUCTION_BASE = r"""<OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism to the variable given the above evaluation output.
Our only goal is to improve the above metric, and nothing else. </OBJECTIVE_FUNCTION>"""


OBJECTIVE_INSTRUCTION_CHAIN = r"""This conversation is part of a larger system. The <OUTPUT_OF_FUNCTION> was later used as {{response_desc}}.

<OBJECTIVE_FUNCTION>Your goal is to give feedback to the variable to address the following feedback on the OUTPUT_OF_FUNCTION: {{response_gradient}} </OBJECTIVE_FUNCTION>"""


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
        backward_engine: "BackwardEngine" = None,
        model_client: ModelClient = None,
        model_kwargs: Dict[str, object] = None,
    ):
        super().__init__()
        self.eval_fn = eval_fn
        self.eval_fn_desc = eval_fn_desc
        self.set_backward_engine(backward_engine, model_client, model_kwargs)

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

    def set_backward_engine(
        self,
        backward_engine: "BackwardEngine" = None,
        model_client: ModelClient = None,
        model_kwargs: Dict[str, object] = None,
    ):
        self.backward_engine = backward_engine
        if not backward_engine:
            log.info(
                "EvalFnToTextLoss: No backward engine provided. Creating one using model_client and model_kwargs."
            )
            self.backward_engine = BackwardEngine(model_client, model_kwargs)
        else:
            if isinstance(backward_engine, BackwardEngine):
                raise TypeError(
                    "EvalFnToTextLoss: backward_engine must be an instance of BackwardEngine."
                )

    @staticmethod
    def _backward_through_one_predecessor(
        pred: Parameter,
        inputs_string: str,
        response: Parameter,
        eval_fn_desc: str,
        backward_engine: "BackwardEngine",
        is_chain: bool = False,
    ):
        if not pred.requires_opt:
            log.debug(
                f"EvalFnToTextLoss: Skipping {pred} as it does not require optimization."
            )
            return
        log.debug(f"EvalFnToTextLoss: Backward through {pred}, is_chain: {is_chain}")

        instruction_str, objective_str = None, None

        # construct the prompt, including three sections
        conversation_str = Prompt(
            CONVERSATION_TEMPLATE_STRING,
            prompt_kwargs={
                "input_str": inputs_string,
                "eval_fn_desc": eval_fn_desc,
                "response_value": response.get_short_value(),
            },
        )()

        if is_chain:
            instruction_str = Prompt(
                CONVERSATION_START_INSTRUCTION_STRING_FN_CHAIN,
                prompt_kwargs={
                    "variable_desc": pred.role_desc,
                    "conversation": conversation_str,
                },
            )()
            objective_str = Prompt(
                OBJECTIVE_INSTRUCTION_CHAIN,
                prompt_kwargs={
                    "response_desc": response.role_desc,
                    "response_gradient": response.get_short_value(),
                },
            )()
        else:
            instruction_str = CONVERSATION_START_INSTRUCTION_STRING_FN_BASE
            objective_str = OBJECTIVE_INSTRUCTION_BASE

        # Compute the gradient
        gradient_value = backward_engine(
            prompt_kwargs={
                "conversation_sec": instruction_str,
                "objective_instruction_sec": objective_str,
                "evaluate_variable_instruction_sec": EVALUATE_VARIABLE_INSTRUCTION,
            }
        )
        log.debug(f"EvalFnToTextLoss: Gradient for {pred}: {gradient_value}")
        gradient_param = Parameter(
            data=gradient_value,
            requires_opt=True,
            role_desc=f"Feedback for {pred.role_desc}",
        )
        pred.gradients.add(gradient_param)
        pred.gradients_context[gradient_param] = {
            "context": conversation_str,
            "response_desc": response.role_desc,
            "variable_desc": pred.role_desc,
        }

        # TODO: reduce meta

    def backward(
        self, response: Parameter, eval_fn_desc: str, kwargs: Dict[str, Parameter]
    ):
        log.info(f"EvalFnToTextLoss: Backward: {response}")
        children_params = response.predecessors
        is_chain = True
        if response.get_gradient_and_context_text.strip() == "":
            log.info(f"EvalFnToTextLoss: Backward: No gradient found for {response}.")
            is_chain = False

        # Convert all input arguments to string
        inputs_string = "\n\n".join(
            [
                f"**{k.replace('_', ' ').capitalize()}(role: {v.role_desc()})**: {v.get_short_value()}"
                for k, v in kwargs.items()
            ]
        )

        # go through all child parameters
        for pred in children_params:
            if pred.requires_opt:
                continue
            self._backward_through_one_predecessor(
                pred,
                inputs_string,
                response,
                eval_fn_desc,
                self.backward_engine,
                is_chain,
            )
