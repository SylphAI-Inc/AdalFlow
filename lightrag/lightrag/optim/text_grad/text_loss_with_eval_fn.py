"""Adapted from text_grad's String Based Function"""

from typing import Callable, Dict, Union, TYPE_CHECKING
import logging
from lightrag.optim.text_grad.function import BackwardContext, GradFunction


if TYPE_CHECKING:
    from lightrag.core import ModelClient

    from lightrag.core.generator import BackwardEngine
from lightrag.core.types import GeneratorOutput
from lightrag.core.component import Component
from lightrag.optim.parameter import Parameter, GradientContext

# from lightrag.optim.text_grad.backend_engine_prompt import EVALUATE_VARIABLE_INSTRUCTION
from lightrag.core.prompt_builder import Prompt
from lightrag.eval.base import BaseEvaluator


log = logging.getLogger(__name__)

###  First part of the user prompt in the backward engine:  Eval function context
CONVERSATION_TEMPLATE_STRING = r"""Eval Function Description: {{eval_fn_desc}}
<START_OF_INPUTS_TO_FUNCTION> {{input_str}} <END_OF_INPUTS_TO_FUNCTION>
<START_OF_OUTPUT_OF_FUNCTION> {{response_value}} <END_OF_OUTPUT_OF_FUNCTION>"""


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
class EvalFnToTextLoss(Component, GradFunction):
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
        model_client: "ModelClient" = None,
        model_kwargs: Dict[str, object] = None,
    ):
        from lightrag.core.generator import BackwardEngine

        super().__init__()
        self.eval_fn = eval_fn
        self.eval_fn_desc = eval_fn_desc
        self.name = f"{self.__class__.__name__}"

        if backward_engine is None:
            log.info(
                "EvalFnToTextLoss: No backward engine provided. Creating one using model_client and model_kwargs."
            )
            if model_client is None or model_kwargs is None:
                raise ValueError(
                    "EvalFnToTextLoss: model_client and model_kwargs must be provided if backward_engine is not provided."
                )
            self.set_backward_engine(backward_engine, model_client, model_kwargs)
        else:
            if not isinstance(backward_engine, BackwardEngine):
                raise TypeError(
                    "EvalFnToTextLoss: backward_engine must be an instance of BackwardEngine."
                )
            self.backward_engine = backward_engine

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, kwargs: Dict[str, Parameter], response_desc: str = None
    ) -> Parameter:
        if response_desc is None:
            response_desc = (
                f"Output of EvalFnToTextLoss with eval_fn_desc: {self.eval_fn_desc}"
            )

        # validate the type of kwargs
        for k, v in kwargs.items():
            if not isinstance(v, Parameter):
                raise TypeError(
                    f"EvalFnToTextLoss: Input argument {k} must be an instance of Parameter."
                )

        response: str = self.eval_fn(**kwargs)

        # Create a parameter
        response: Parameter = Parameter(
            alias=self.name + "_output",
            data=response,
            requires_opt=True,
            predecessors=list(kwargs.values()),
            role_desc=response_desc,
        )

        log.info(f"EvalFnToTextLoss: Input: {kwargs}, Output: {response}")
        response.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward,
                backward_engine=self.backward_engine,
                response=response,
                eval_fn_desc=self.eval_fn_desc,
                kwargs=kwargs,
            )
        )
        return response

    def set_backward_engine(
        self,
        backward_engine: "BackwardEngine" = None,
        model_client: "ModelClient" = None,
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
                "response_value": response.data,
            },
        )()

        conv_ins_template = CONVERSATION_START_INSTRUCTION_STRING_FN_BASE
        obj_ins_template = OBJECTIVE_INSTRUCTION_BASE

        if is_chain:
            conv_ins_template = CONVERSATION_START_INSTRUCTION_STRING_FN_CHAIN
            obj_ins_template = OBJECTIVE_INSTRUCTION_CHAIN

        instruction_str = Prompt(
            conv_ins_template,
            prompt_kwargs={
                "variable_desc": pred.role_desc,
                "conversation": conversation_str,
            },
        )()
        objective_str = Prompt(
            obj_ins_template,
            prompt_kwargs={
                "response_desc": response.role_desc,
                "response_gradient": response.data,
            },
        )()

        log.info(f"EvalFnToTextLoss: Instruction: {instruction_str}")
        log.info(f"EvalFnToTextLoss: Objective: {objective_str}")
        log.info(f"EvalFnToTextLoss: Conversation: {conversation_str}")

        # Compute the gradient
        backward_engine_prompt_kwargs = {
            "conversation_sec": instruction_str,
            "objective_instruction_sec": objective_str,
            # "evaluate_variable_instruction_sec": eval_str,
        }
        gradient_value: GeneratorOutput = backward_engine(
            prompt_kwargs=backward_engine_prompt_kwargs
        )
        gradient_prompt = backward_engine.get_prompt(**backward_engine_prompt_kwargs)
        gradient_value_data = (
            gradient_value.data
            or backward_engine.failure_message_to_optimizer(
                gradient_response=gradient_value
            )
        )

        log.debug(f"EvalFnToTextLoss: Gradient for {pred}: {gradient_value_data}")
        gradient_param = Parameter(
            alias=f"{response.alias}_to_{pred.alias}_grad",
            data=gradient_value_data,
            requires_opt=True,
            gradient_prompt=gradient_prompt,
            role_desc=f"Feedback for {pred.role_desc}",
        )
        pred.gradients.add(gradient_param)
        pred.gradients_context[gradient_param] = GradientContext(
            context=conversation_str,
            response_desc=response.role_desc,
            variable_desc=pred.role_desc,
        )

        # TODO: reduce meta

    def backward(
        self,
        response: Parameter,
        eval_fn_desc: str,
        kwargs: Dict[str, Parameter],
        backward_engine: "BackwardEngine",
    ):
        log.info(f"EvalFnToTextLoss: Backward: {response}")
        children_params = response.predecessors
        is_chain = True
        if response.get_gradient_and_context_text().strip() == "":
            log.info(f"EvalFnToTextLoss: Backward: No gradient found for {response}.")
            is_chain = False

        # Convert all input arguments to string
        inputs_string = "\n\n".join(
            [f"{k}(role: {v.role_desc}): {v.data}" for k, v in kwargs.items()]
        )

        # go through all child parameters
        for pred in children_params:
            if not pred.requires_opt:
                log.debug(
                    f"EvalFnToTextLoss: Skipping {pred} as it does not require optimization."
                )
                continue
            self._backward_through_one_predecessor(
                pred,
                inputs_string,
                response,
                eval_fn_desc,
                backward_engine,
                is_chain,
            )


if __name__ == "__main__":
    # Example of using EvalFnToTextLoss
    from lightrag.utils import setup_env, get_logger
    from lightrag.eval.answer_match_acc import AnswerMatchAcc
    from lightrag.components.model_client import OpenAIClient  # dir: model_client
    from lightrag.core.generator import Generator, BackwardEngine
    from lightrag.core.component import fun_to_component

    logger = get_logger(level="DEBUG", filename="lib_text_grad.log")

    setup_env()

    gpt_3_model = {
        "model_client": OpenAIClient(),
        "model_kwargs": {
            "model": "gpt-3.5-turbo",
        },
    }
    gpt_4o_model = {
        "model_client": OpenAIClient(),
        "model_kwargs": {
            "model": "gpt-4o",
        },
    }

    @fun_to_component
    def parse_integer_answer(answer: str, only_first_line: bool = False):
        try:
            if only_first_line:
                answer = answer.strip().split("\n")[0]
            answer = answer.strip()
            # find the last token that has a number in it
            answer = [
                token for token in answer.split() if any(c.isdigit() for c in token)
            ][-1]
            answer = answer.split(".")[0]
            answer = "".join([c for c in answer if c.isdigit()])
            answer = int(answer)

        except (ValueError, IndexError):
            answer = 0

        return answer

    evaluator = AnswerMatchAcc()
    eval_fn_desc = "Answer Match Accuracy"
    single_compute_eval = AnswerMatchAcc().compute_single_item

    backward_engine = BackwardEngine(**gpt_4o_model)
    # backward_engine.set_mock_output(mock_output_data="1")

    eval_fn_to_text_loss = EvalFnToTextLoss(
        eval_fn=single_compute_eval,
        eval_fn_desc=eval_fn_desc,
        backward_engine=backward_engine,
    )
    x = Parameter(
        alias="x",
        data="I have a cauliflower, a stalk of celery, a cabbage, and a garlic. How many vegetables do I have?",
        requires_opt=False,
        role_desc="The question to the language model",
    )

    system_prompt = Parameter(
        alias="system_prompt",
        data="You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
        requires_opt=True,
        role_desc="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task",
    )

    model = Generator(
        prompt_kwargs={"task_desc_str": system_prompt},
        **gpt_3_model,
        output_processors=parse_integer_answer,
    )
    # model.set_mock_output(mock_output_data="4")
    model.train()
    print(f"model.train: {model.training}")

    y: Parameter = model(prompt_kwargs={"input_str": x})
    print(f"y: {y}")

    loss = eval_fn_to_text_loss(
        {
            "y": y,
            "y_gt": Parameter(
                alias="y_gt",
                data="4",
                requires_opt=False,
                role_desc="Correct answer",
            ),
        }
    )
    print(f"loss: {loss}")
    loss.backward()
    print(loss.to_dict())
    assert len(loss.predecessors) == 2
    assert len(y.predecessors) == 2
    dot = loss.draw_graph(add_grads=True, filepath="real_data")
    # print("dot: ", dot)

#     Variable(data=1, requires_opt=True, role_desc=Output of the string-based function with purpose:
#              The runtime of string-based function that checks if the prediction is correct.,
#              predecessors={Variable(data=4, requires_opt=False, role_desc=correct answer for the query,
#                                     predecessors=set(), gradients=set()),
#                                     Variable(data=To determine the number of vegetables,
#                                              we need to count each individual item.
#                                                The cauliflower, celery, cabbage,
#                                                and garlic are all separate vegetables.
#                                                Therefore, you have 4 vegetables in total.

# Answer: 4, requires_opt=True, role_desc=response from the language model,
# predecessors={Variable(data=I have a cauliflower, a stalk of celery, a cabbage, and a garlic. How many vegetables do I have?, requires_opt=False, role_desc=query to the language model, predecessors=set(), gradients=set()), Variable(data=You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value., requires_opt=True, role_desc=structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task, predecessors=set(), gradients=set())}, gradients=set())}, gradients=set())

# loss: Parameter(alias=None, data=1.0, requires_opt=True, role_desc=Output of EvalFnToTextLoss with eval_fn_desc: Answer Match Accuracy,
# predecessors={Parameter(alias=None, data=1, requires_opt=False, role_desc=Predicted answer, predecessors=set(), gradients=set()),
# Parameter(alias=None, data=1, requires_opt=False, role_desc=Correct answer, predecessors=set(), gradients=set())}, gradients=set())
# {'alias': None, 'data': 1.0, 'requires_opt': True, 'role_desc': 'Output of EvalFnToTextLoss with eval_fn_desc: Answer Match Accuracy', 'predecessors': [{'alias': None, 'data': '1', 'requires_opt': False, 'role_desc': 'Predicted answer', 'predecessors': [], 'gradients': [], 'proposed_data': None, 'gradients_context': [], 'grad_fn': 'None'}, {'alias': None, 'data': '1', 'requires_opt': False, 'role_desc': 'Correct answer', 'predecessors': [], 'gradients': [], 'proposed_data': None, 'gradients_context': [], 'grad_fn': 'None'}], 'gradients': [], 'proposed_data': None, 'gradients_context': [], 'grad_fn': '__main__.EvalFnToTextLoss.backward'}
