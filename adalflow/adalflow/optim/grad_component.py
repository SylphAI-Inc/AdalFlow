"""Base class for Autograd Components that can be called and backpropagated through."""

from typing import TYPE_CHECKING, Callable, Optional, Dict
from collections import OrderedDict
import uuid
import logging
from copy import deepcopy

if TYPE_CHECKING:
    from adalflow.core.generator import BackwardEngine

    from adalflow.core import ModelClient
from adalflow.optim.parameter import (
    Parameter,
    OutputParameter,
)
from adalflow.optim.gradient import Gradient, GradientContext
from adalflow.optim.types import ParameterType
from adalflow.core.types import GeneratorOutput
from adalflow.utils import printc

import json

from adalflow.core.component import Component
from adalflow.optim.function import BackwardContext
from adalflow.core.prompt_builder import Prompt
from adalflow.optim.text_grad.backend_engine_prompt import (
    GRAD_COMPONENT_CONVERSATION_TEMPLATE_STRING,
    LOSS_CONVERSATION_START_INSTRUCTION_STRING_FN,
    OBJECTIVE_INSTRUCTION_BASE,
    OBJECTIVE_INSTRUCTION_CHAIN,
)


__all__ = ["GradComponent", "FunGradComponent", "fun_to_grad_component"]

log = logging.getLogger(__name__)


class GradComponent(Component):
    __doc__ = """A base class to define interfaces for an auto-grad component/operator.

    Compared with `Component`, `GradComponent` defines three important interfaces:
    - `forward`: the forward pass of the function, returns a `Parameter` object that can be traced and backpropagated.
    - `backward`: the backward pass of the function, updates the gradients/prediction score backpropagated from a "loss" parameter.
    - `set_backward_engine`: set the backward engine(a form of generator) to the component, which is used to backpropagate the gradients using LLM.

    The __call__ method will check if the component is in training mode,
    and call the `forward` method to return a `Parameter` object if it is in training mode,
    otherwise, it will call the `call` method to return the output such as "GeneratorOutput", "RetrieverOutput", etc.

    Note: Avoid using the attributes and methods that are defined here and in the `Component` class unless you are overriding them.
    """

    backward_engine: "BackwardEngine"
    _component_type = "grad"
    id = None
    _component_desc = "GradComponent"
    _disable_backward_engine = False

    def __init__(
        self,
        desc: str,  # what is this component for? Useful for training.
        name: Optional[str] = None,
        backward_engine: Optional["BackwardEngine"] = None,
        model_client: "ModelClient" = None,
        model_kwargs: Dict[str, object] = None,
    ):
        super().__init__()
        super().__setattr__("id", str(uuid.uuid4()))

        self.desc = desc
        self.backward_engine = backward_engine
        self.model_client = model_client
        self.name = name or f"{self.__class__.__name__}"

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

    def disable_backward_engine(self):
        r"""Does not run gradients generation, but still with backward to gain module-context"""
        self._disable_backward_engine = True

    def call(self, *args, **kwargs):
        raise NotImplementedError("call method is not implemented")

    async def acall(self, *args, **kwargs):
        r"""Implement this for your async call."""
        raise NotImplementedError("acall method is not implemented")

    def forward(self, *args, **kwargs) -> "Parameter":
        r"""Default forward method for training:
        1. for all args and kwargs, if it is a `Parameter` object, it will be tracked as `Predecessor`.
        2. Trace input_args and full_response in the parameter object.
        3. Return the parameter object.
        """

        from adalflow.optim.parameter import Parameter, OutputParameter

        log.debug(
            f"Forwarding through {self.name} with args: {args} and kwargs: {kwargs}"
        )

        # 1. get all predecessors from all args and kwargs
        input_args = OrderedDict()

        # Add positional args to the ordered dict
        for idx, arg in enumerate(args):
            input_args[f"arg_{idx}"] = arg

        # Get data id from the kwargs
        data_id = kwargs.get("id", None)

        # Add keyword args to the ordered dict, preserving order
        predecessors = []
        for v in input_args.values():
            if isinstance(v, Parameter):
                predecessors.append(v)
                if v.param_type == ParameterType.INPUT:
                    v.data_id = kwargs.get("id", None)
                if data_id is None:
                    data_id = v.data_id
        # printc(f"kwargs: {kwargs}")
        # discard_keys = []
        for k, v in kwargs.items():
            if isinstance(v, Parameter):
                predecessors.append(v)
                if v.param_type == ParameterType.INPUT:
                    v.data_id = kwargs.get("id", None)
                if data_id is None:
                    data_id = v.data_id
            # support list of Parameters by flattening them
            elif isinstance(v, list):
                for i, p in enumerate(v):
                    if isinstance(p, Parameter):
                        predecessors.append(p)

        # 2. unwrap the parameter object to take only the data, successor_map_fn: lambda x: x.data in default
        # unwrap args
        unwrapped_args = []
        for k, v in input_args.items():
            if isinstance(v, Parameter):
                unwrapped_args.append(v.map_to_successor(self))
            else:
                unwrapped_args.append(v)

        unwrapped_kwargs = {}
        # unwrap kwargs
        for k, v in kwargs.items():
            if isinstance(v, Parameter):
                unwrapped_kwargs[k] = v.map_to_successor(self)
            elif isinstance(v, list):
                values = []
                for p in v:
                    if isinstance(p, Parameter):
                        values.append(p.map_to_successor(self))
                    else:
                        values.append(p)
                unwrapped_kwargs[k] = values
            else:
                unwrapped_kwargs[k] = v

        # 3. call the function with unwrapped args and kwargs
        unwrapped_args = tuple(unwrapped_args)

        log.debug(f"Unwrapped args: {unwrapped_args}")
        log.debug(f"Unwrapped kwargs: {unwrapped_kwargs}")

        call_response = self.call(*unwrapped_args, **unwrapped_kwargs)

        if isinstance(call_response, Parameter):
            raise ValueError(
                f"A GradComponent call should not return Parameter, got {call_response.name}"
            )
            predecessors.append(call_response)
            return call_response

        # 4. Create a Parameter object to trace the forward pass
        # use unwrapped args  and unwrapped kwargs to trace the forward pass
        tracing_args = {i: v for i, v in enumerate(unwrapped_args)}
        tracing_args.update(**unwrapped_kwargs)

        response = OutputParameter(
            data=call_response,
            name=self.name + "_output",
            role_desc=self.name + " response",
            param_type=ParameterType.OUTPUT,
            data_id=data_id,
        )
        response.set_predecessors(predecessors)
        response.trace_forward_pass(
            input_args=tracing_args,
            full_response=call_response,
            id=self.id,  # this is component id
            name=self.name,
        )
        response.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward,
                response=response,
                id=data_id,
                input_kwargs=kwargs,
                disable_backward_engine=self._disable_backward_engine,
            )
        )
        return response

    def backward_with_pass_through_gradients(
        self, *, response: "Parameter", id: str = None, **kwargs
    ):
        """Pass-through gradient to the predecessors.

        Backward pass of the function. In default, it will pass all the scores to the predecessors.

        Note: backward is mainly used internally and better to only allow kwargs as the input.

        Subclass should implement this method if you need additional backward logic.
        """

        log.info(f"GradComponent backward: {response.name}")
        children_params = response.predecessors

        if response.get_gradient_and_context_text().strip() == "":
            log.info(f"Generator: Backward: No gradient found for {response}.")

        for _, pred in enumerate(children_params):
            if response.score is not None:
                pred.set_score(response.score)

            if pred.param_type == ParameterType.DEMOS:
                pred.add_score_to_trace(
                    trace_id=id, score=response.score, is_teacher=self.teacher_mode
                )
            for grad in response.gradients:
                # NOTE: make a copy of the gradient, we should not modify the original gradient
                grad = deepcopy(grad)

                grad.is_default_copy = (
                    True  # response and pred will keep the original gradient
                )
                pred.add_gradient(grad)

    @staticmethod
    def _backward_through_one_predecessor(
        pred: Parameter,
        kwargs: Dict[str, Parameter],
        response: Parameter,
        desc: str,
        backward_engine: "BackwardEngine",
        ground_truth: object = None,
        is_intermediate_node: bool = False,  # if the node is an intermediate node in the backpropagation chain
        metadata: Dict[str, str] = None,
        disable_backward_engine: bool = False,
    ):
        if not pred.requires_opt:
            if response.score is not None:
                pred.set_score(response.score)
            log.debug(
                f"EvalFnToTextLoss: Skipping {pred} as it does not require optimization."
            )
            return
        log.debug(
            f"EvalFnToTextLoss: Backward through {pred}, is_intermediate_node: {is_intermediate_node}"
        )

        if pred.check_if_already_computed_gradient_respect_to(response.id):
            log.info(
                f"EvalFnToTextLoss: Gradient already computed for {pred.role_desc} with respect to {response.role_desc}"
            )

            return

        if backward_engine is None:
            log.error(
                "EvalFnToTextLoss: backward_engine is required for text prompt optimization."
            )
            raise ValueError(
                "EvalFnToTextLoss: backward_engine is required for text prompt optimization."
            )

        instruction_str, objective_str = None, None

        # convert kwargs to key, (value, type(eval_input))

        inputs = {}

        for k, v in kwargs.items():
            if isinstance(v, Parameter):
                inputs[k] = (v.get_param_info(), str(type(v.eval_input)))
            elif isinstance(v, list):
                # flat the list to multiple parameters

                for i, p in enumerate(v):
                    if isinstance(p, Parameter):
                        flat_key = f"{k}_{i}"
                        inputs[flat_key] = (p.get_param_info(), str(type(p.eval_input)))

        # response information
        conversation_str = Prompt(
            GRAD_COMPONENT_CONVERSATION_TEMPLATE_STRING,
            prompt_kwargs={
                "inputs": inputs,
                "component_desc": desc,
                "response_value": response.get_prompt_data(),
                "metadata": json.dumps(metadata) if metadata else None,
            },
        )()

        conv_ins_template = LOSS_CONVERSATION_START_INSTRUCTION_STRING_FN
        obj_ins_template = OBJECTIVE_INSTRUCTION_BASE

        if is_intermediate_node:
            printc(f"is_intermediate_node: {is_intermediate_node}")
            # conv_ins_template = CONVERSATION_START_INSTRUCTION_STRING_FN_CHAIN
            obj_ins_template = OBJECTIVE_INSTRUCTION_CHAIN

        instruction_str = Prompt(
            conv_ins_template,
            prompt_kwargs={
                "variable": pred.get_param_info(),
                "conversation_str": conversation_str,
            },
        )()
        response_gradient = response.get_gradients_str()
        # response_gradient = response.get_gradients_component_schema()
        if not response_gradient:
            raise ValueError(
                f"Generator: No gradient found for {response}. Please check the response. pred: {pred}"
            )
        objective_str = Prompt(
            obj_ins_template,
            prompt_kwargs={
                "response_name": response.name,
                "response_desc": response.role_desc,
                "response_gradient": response_gradient,
            },
        )()

        log.info(f"EvalFnToTextLoss: Instruction: {instruction_str}")
        log.info(f"EvalFnToTextLoss: Objective: {objective_str}")
        log.info(f"EvalFnToTextLoss: Conversation: {conversation_str}")

        # Compute the gradient
        backward_engine_prompt_kwargs = {
            "conversation_sec": instruction_str,
            "objective_instruction_sec": objective_str,
        }
        gradient_value_data = None
        if not disable_backward_engine:
            gradient_value: GeneratorOutput = backward_engine(
                prompt_kwargs=backward_engine_prompt_kwargs
            )
            gradient_prompt = backward_engine.get_prompt(
                **backward_engine_prompt_kwargs
            )
            gradient_value_data = (
                gradient_value.data
                or backward_engine.failure_message_to_optimizer(
                    gradient_response=gradient_value
                )
            )

            gradient_value_data = (
                f"expected answer: {ground_truth},\n Feedback: {gradient_value_data}"
            )

            log.debug(f"EvalFnToTextLoss: Gradient for {pred}: {gradient_value_data}")

        gradient_param = Gradient(
            data=gradient_value_data,
            data_id=response.data_id,
            score=response.score,
            from_response=response,
            to_pred=pred,
        )
        gradient_param.add_prompt(gradient_prompt)
        gradient_param.add_context(
            GradientContext(
                input_output=conversation_str,
                response_desc=response.role_desc,
                variable_desc=pred.role_desc,
            )
        )
        pred.add_gradient(gradient_param)

        if response.score is not None:
            pred.set_score(response.score)
        pred.set_gt(ground_truth)

    def backward(
        self,
        *,
        response: "OutputParameter",
        id: str = None,
        disable_backward_engine=False,
        **kwargs,
    ):
        """Backward pass of the function. In default, it will pass all the scores to the predecessors.

        Note: backward is mainly used internally and better to only allow kwargs as the input.

        Subclass should implement this method if you need additional backward logic.
        """

        log.info(f"GradComponent backward: {response.name}")
        children_params = response.predecessors

        input_kwargs = kwargs.get("input_kwargs", {})

        is_intermediate_node = False
        response_gradient_context = response.get_gradient_and_context_text().strip()
        if response_gradient_context != "":
            log.info("EvalFnToTextLoss is an intermediate node.")
            is_intermediate_node = True

        if response.get_gradient_and_context_text().strip() == "":
            log.info(f"Generator: Backward: No gradient found for {response}.")

        # use pass through gradient when there is one predecessor
        if not self.backward_engine or len(children_params) < 2:
            self.backward_with_pass_through_gradients(response=response, id=id)

        else:

            for _, pred in enumerate(children_params):
                if response.score is not None:
                    pred.set_score(response.score)
                printc(f"score {response.score} for pred name: {pred.name}")
                if not pred.requires_opt:
                    continue

                if pred.param_type == ParameterType.DEMOS:
                    pred.add_score_to_trace(
                        trace_id=id, score=response.score, is_teacher=self.teacher_mode
                    )

                printc(f"pred: {pred.name}, response: {response.name}")

                self._backward_through_one_predecessor(
                    pred=pred,
                    kwargs=input_kwargs,
                    response=response,
                    backward_engine=self.backward_engine,
                    desc=self.desc,
                    is_intermediate_node=is_intermediate_node,
                    disable_backward_engine=disable_backward_engine,
                )


class FunGradComponent(GradComponent):

    def __init__(
        self,
        fun: Callable,
        # afun: Optional[Callable] = None,
        desc: str = "",
        doc_string=None,
    ):

        desc = desc or fun.__doc__ or f"Function: {fun.__name__}"

        super().__init__(desc=desc, name=fun.__name__)
        self.fun_name = fun.__name__
        self.fun = fun
        # set the docstring
        self.doc_string = doc_string
        setattr(
            self.fun,
            "__doc__",
            doc_string or fun.__doc__ or f"Function: {fun.__name__}",
        )

        setattr(self.fun, "__name__", fun.__name__)

    def call(self, *args, **kwargs):

        kwargs.pop("doc_string", None)

        return self.fun(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Parameter:
        """add func_doc_string to the kwargs before calling the super().forward"""
        kwargs["doc_string"] = self.doc_string
        output = super().forward(*args, **kwargs)
        return output

    def _extra_repr(self) -> str:
        return (
            super()._extra_repr()
            + f"fun_name={self.fun_name}, fun={self.fun.__name__}, fun_doc={self.fun.__doc__}"
        )


def fun_to_grad_component(desc: str = "", doc_string=None) -> Callable:
    """
    Return a decorator that, when applied to a function `fun`,
    wraps it in a GradComponent with the given `desc`.

    Examples:

    1. As a decorator:


    ::code-block :: python

        @fun_to_grad_component(desc="This is a test function", doc_string=Parameter(
            data="Finish the task with verbatim short factoid responses from retrieved context.",
            param_type=ParameterType.PROMPT,
            requires_opt=True,
            role_desc="Instruct how the agent creates the final answer from the step history.",
        ))
        def my_function(x):
            return x + 1

        print(my_function(1))

    2. As a function:

    ::code-block :: python

        def my_function(x):
            return x + 1

        my_function_component = fun_to_grad_component(desc="This is a test function")(my_function)
    """

    def decorator(fun):
        # 1) build the class name
        class_name = (
            "".join(part.capitalize() for part in fun.__name__.split("_"))
            + "GradComponent"
        )
        # 2) define the new class
        component_class = type(
            class_name,
            (FunGradComponent,),
            {
                "__init__": lambda self: FunGradComponent.__init__(
                    self, fun=fun, desc=desc, doc_string=doc_string
                )
            },
        )
        return component_class()

    return decorator


if __name__ == "__main__":
    # Test FunGradComponent
    from adalflow.optim.parameter import Parameter

    def my_function(x):
        __doc__ = Parameter(  # noqa F841
            data="Finish the task with verbatim short factoid responses from retrieved context.",
            param_type=ParameterType.PROMPT,
            requires_opt=True,
            role_desc="Instruct how the agent creates the final answer from the step history.",
        )
        return x + 1

    my_function_component = fun_to_grad_component()(my_function)
    print(my_function_component)  # 2
    # eval mode
    output = my_function_component(1)
    print(output)
    # training mode
    my_function_component.train()
    output = my_function_component(Parameter(data=1, name="input"))
    print(output)

    # now test the decorator
    @fun_to_grad_component(
        desc="This is a test function",
        doc_string=Parameter(
            data="Finish the task with verbatim short factoid responses from retrieved context.",
            param_type=ParameterType.PROMPT,
            requires_opt=True,
            role_desc="Instruct how the agent creates the final answer from the step history.",
        ),
    )
    def my_function(x):

        return x + 1

    print(my_function(1))
    # eval mode
    output = my_function(1)
    print(output)
    assert output == 2

    # training mode
    my_function.train()
    print(my_function)
    output = my_function(Parameter(data=1, name="input"))
    print(output)
