"""Adapted from text_grad's String Based Function"""

from typing import Callable, Dict, Union, TYPE_CHECKING, Optional
import logging
import json
from adalflow.optim.loss_component import LossComponent
from adalflow.optim.function import BackwardContext


if TYPE_CHECKING:
    from adalflow.core import ModelClient
    from adalflow.core.generator import BackwardEngine
from adalflow.core.types import GeneratorOutput
from adalflow.optim.parameter import Parameter, GradientContext
from adalflow.optim.types import ParameterType

from adalflow.core.prompt_builder import Prompt
from adalflow.eval.base import BaseEvaluator


log = logging.getLogger(__name__)

###  Loss/Score Information  ###
CONVERSATION_TEMPLATE_STRING = r"""
The variable is passed to the eval function and compared with a target/ground truth value.

<EVAL_FUNC_DESCRIPTION>: {{eval_fn_desc}}
<INPUTS>: {{input_str}}
<OUTPUTS/SCORE>: {{response_value}}
{% if metadata %}
Note: {{metadata}}
{% endif %}"""


# Does not have gradient on the output, the loss function of the backpropagation chain
# CONVERSATION_START_INSTRUCTION_STRING_FN_BASE = r"""You will give feedback to a variable with the following role:
# <ROLE> {{variable_desc}} </ROLE>.
# Here is an evaluation of the variable using the eval function:
# {{conversation}}"""

# Has the gradient on the output, the layer in the backpropagation chain
# Conversation will be provided differently.

### Variable Information ###
CONVERSATION_START_INSTRUCTION_STRING_FN = r"""
TARGET VARIABLE:
<NAME>: {{variable_name}}
<ROLE> {{variable_desc}} </ROLE>
<VARIABLE> {{variable_value}} </VARIABLE>
{{conversation_str}}
"""

# Third part of the user prompt
OBJECTIVE_INSTRUCTION_BASE = r"""<OBJECTIVE_FUNCTION>
Your only goal is to clearly states how it obtained the "<OUTPUTS/SCORE>".
Especially when the score is low.
Be CONCISE.
If you have enough context, add a more specific feedback on how it failed.
</OBJECTIVE_FUNCTION>"""


OBJECTIVE_INSTRUCTION_CHAIN = r"""This conversation is part of a larger system. The <INPUTS/SCORE> was later used as "{{response_name}}: {{response_desc}}".
<OBJECTIVE_FUNCTION>
Your only goal is to clearly states how it obtained the "Eval output/score": {{response_gradient}}.
Especially when the score is low.
Be CONCISE.
If you have enough context, add a more specific feedback on how it failed.
</OBJECTIVE_FUNCTION>"""


class EvalFnToTextLoss(LossComponent):
    __doc__ = """Convert an evaluation function to a text loss.

    LossComponent will take an eval function and output a score
    (usually a float in range [0, 1], and the higher the better, unlike the loss function in model training).

    In math:

    score/loss = eval_fn(y_pred, y_gt)

    The gradident/feedback = d(score)/d(y_pred) will be computed using a backward engine.
    Gradient_context = GradientContext(
        context=conversation_str,
        response_desc=response.role_desc,
        variable_desc=role_desc,
    )

    Args:
        eval_fn: The evaluation function that takes a pair of y and y_gt and returns a score.
        eval_fn_desc: Description of the evaluation function.
        backward_engine: The backward engine to use for the text prompt optimization.
        model_client: The model client to use for the backward engine if backward_engine is not provided.
        model_kwargs: The model kwargs to use for the backward engine if backward_engine is not provided.

    """

    def __init__(
        self,
        eval_fn: Union[Callable, BaseEvaluator],
        eval_fn_desc: str,
        backward_engine: Optional["BackwardEngine"] = None,
        model_client: "ModelClient" = None,
        model_kwargs: Dict[str, object] = None,
    ):
        from adalflow.core.generator import BackwardEngine

        super().__init__()
        self.eval_fn = eval_fn
        self.eval_fn_desc = eval_fn_desc
        self.name = f"{self.__class__.__name__}"

        self.backward_engine = None

        if backward_engine is None:
            log.info(
                "EvalFnToTextLoss: No backward engine provided. Creating one using model_client and model_kwargs."
            )
            if model_client and model_kwargs:

                self.set_backward_engine(backward_engine, model_client, model_kwargs)
        else:
            if not isinstance(backward_engine, BackwardEngine):
                raise TypeError(
                    "EvalFnToTextLoss: backward_engine must be an instance of BackwardEngine."
                )
            self.backward_engine = backward_engine

    # def __call__(self, *args, **kwargs) -> Parameter:
    #     r"""Different from default GradComponent __call__.
    #     Only Parameter as this is not needed in a normal task pipeline, but only
    #     for training generator."""
    #     return self.forward(*args, **kwargs)

    def forward(
        self,
        kwargs: Dict[str, Parameter],
        response_desc: str = None,
        metadata: Dict[str, str] = None,  # additional notes on the input kwargs
    ) -> Parameter:
        if response_desc is None:
            response_desc = "Output of EvalFnToTextLoss."

        # validate the type of kwargs
        predesessors = []
        for k, v in kwargs.items():
            if not isinstance(v, Parameter):
                raise TypeError(
                    f"EvalFnToTextLoss: All inputs must be Parameters. Got {type(v)} for {k}."
                )
            if isinstance(v, Parameter):
                predesessors.append(v)
        eval_inputs = {}
        for k, v in kwargs.items():
            eval_inputs[k] = v.eval_input
        score: float = self.eval_fn(**eval_inputs)

        # Create a parameter
        # TODO: improve the readability of the input and response
        eval_param: Parameter = Parameter(
            name=self.name + "_output",
            data=score,
            requires_opt=True,
            role_desc=response_desc,
            score=score,
        )
        eval_param.set_predecessors(predesessors)

        log.info(f"EvalFnToTextLoss: Input: {kwargs}, Output: {eval_param}")
        eval_param.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward,
                backward_engine=self.backward_engine,
                response=eval_param,
                eval_fn_desc=self.eval_fn_desc,
                kwargs=kwargs,
                metadata=metadata,
            )
        )
        return eval_param

    def set_backward_engine(
        self,
        backward_engine: "BackwardEngine" = None,
        model_client: "ModelClient" = None,
        model_kwargs: Dict[str, object] = None,
    ):
        from adalflow.core.generator import BackwardEngine

        self.backward_engine = backward_engine
        if not backward_engine:
            log.info(
                "EvalFnToTextLoss: No backward engine provided. Creating one using model_client and model_kwargs."
            )
            self.backward_engine = BackwardEngine(model_client, model_kwargs)
        else:
            if type(backward_engine) is not BackwardEngine:
                raise TypeError(
                    f"EvalFnToTextLoss: backward_engine must be an instance of BackwardEngine. Got {type(backward_engine)}."
                )

    @staticmethod
    def _backward_through_one_predecessor(
        pred: Parameter,
        inputs_string: str,
        response: Parameter,
        eval_fn_desc: str,
        backward_engine: "BackwardEngine",
        is_chain: bool = False,
        metadata: Dict[str, str] = None,
    ):
        if not pred.requires_opt:
            log.debug(
                f"EvalFnToTextLoss: Skipping {pred} as it does not require optimization."
            )
            return
        log.debug(f"EvalFnToTextLoss: Backward through {pred}, is_chain: {is_chain}")

        if pred.check_if_already_computed_gradient_respect_to(response.id):
            log.info(
                f"EvalFnToTextLoss: Gradient already computed for {pred.role_desc} with respect to {response.role_desc}"
            )
            # print(
            #     f"Gradient already computed for {pred.role_desc} with respect to {response.role_desc}"
            # )
            return

        if backward_engine is None:
            log.error(
                "EvalFnToTextLoss: backward_engine is required for text prompt optimization."
            )
            raise ValueError(
                "EvalFnToTextLoss: backward_engine is required for text prompt optimization."
            )

        instruction_str, objective_str = None, None

        # response information
        conversation_str = Prompt(
            CONVERSATION_TEMPLATE_STRING,
            prompt_kwargs={
                "input_str": inputs_string,
                "eval_fn_desc": eval_fn_desc,
                "response_value": response.data,
                "metadata": json.dumps(metadata) if metadata else None,
            },
        )()

        conv_ins_template = CONVERSATION_START_INSTRUCTION_STRING_FN
        obj_ins_template = OBJECTIVE_INSTRUCTION_BASE

        if is_chain:
            # conv_ins_template = CONVERSATION_START_INSTRUCTION_STRING_FN_CHAIN
            obj_ins_template = OBJECTIVE_INSTRUCTION_CHAIN

        instruction_str = Prompt(
            conv_ins_template,
            prompt_kwargs={
                "variable_desc": pred.role_desc,
                "variable_name": pred.name,
                "variable_value": pred.data,
                "conversation_str": conversation_str,
            },
        )()
        objective_str = Prompt(
            obj_ins_template,
            prompt_kwargs={
                "response_name": response.name,
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
        # gradient_prompt = backward_engine.get_prompt(**backward_engine_prompt_kwargs)
        gradient_value_data = (
            gradient_value.data
            or backward_engine.failure_message_to_optimizer(
                gradient_response=gradient_value
            )
        )
        # print(f"gradient_prompt: {gradient_prompt}")
        # gradient_value_data = response.data.to_yaml()

        log.debug(f"EvalFnToTextLoss: Gradient for {pred}: {gradient_value_data}")

        # score should be passed to grad
        gradient_param = Parameter(
            name=f"{response.name}_to_{pred.name}_grad",
            data=gradient_value_data,
            requires_opt=True,
            # gradient_prompt=gradient_prompt,
            role_desc=f"Feedback for {pred.role_desc}",
            score=response.data,
            from_response_id=response.id,
            param_type=ParameterType.GRADIENT,
        )
        pred.add_gradient(gradient_param)
        pred.gradients_context[gradient_param] = GradientContext(
            context=conversation_str,
            response_desc=response.role_desc,
            variable_desc=pred.role_desc,
        )

        # backward the end to end score
        pred.set_score(response.data)
        print(f"setting pred name {pred.name} score to {response.data}")

        # TODO: reduce meta

    def backward(
        self,
        response: Parameter,
        eval_fn_desc: str,
        kwargs: Dict[str, Parameter],
        backward_engine: Optional[
            "BackwardEngine"
        ] = None,  # only needed for text prompt optimization
        metadata: Dict[str, str] = None,
    ):
        r"""Ensure to set backward_engine for the text prompt optimization. It can be None if you
        are only doing demo optimization and it will not have gradients but simply backpropagate the score.
        """
        log.info(f"EvalFnToTextLoss: Backward: {response}")
        children_params = response.predecessors
        is_chain = True
        response_gradient_context = response.get_gradient_and_context_text().strip()
        if response_gradient_context == "":
            log.info(f"EvalFnToTextLoss: Backward: No gradient found for {response}.")
            is_chain = False
        log.info(f"response_gradient_context: {response_gradient_context}")

        # go through all child parameters
        if backward_engine and not response.backward_engine_disabled:
            # Convert all input arguments to string
            inputs_string = "\n\n".join(
                [
                    f"({k}) (role: {v.role_desc}), data: {v.data}, input_to_eval_fn: {v.eval_input}, data_type: {type(v.eval_input)}"
                    for k, v in kwargs.items()
                ]
            )
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
                    metadata,
                )
        # backward for the score for the demo
        for pred in children_params:
            if not pred.requires_opt:
                log.debug(
                    f"EvalFnToTextLoss: Skipping {pred} as it does not require optimization."
                )
                continue
            pred._score = float(response.data)
            log.info(f"setting pred name {pred.name} score to {response.data}")


if __name__ == "__main__":
    # Example of using EvalFnToTextLoss
    from adalflow.utils import setup_env, get_logger
    from adalflow.eval.answer_match_acc import AnswerMatchAcc
    from adalflow.components.model_client import OpenAIClient  # dir: model_client
    from adalflow.core.generator import Generator, BackwardEngine
    from adalflow.core.component import fun_to_component

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
        name="x",
        data="I have a cauliflower, a stalk of celery, a cabbage, and a garlic. How many vegetables do I have?",
        requires_opt=False,
        role_desc="The question to the language model",
    )

    system_prompt = Parameter(
        name="system_prompt",
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
                name="y_gt",
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
