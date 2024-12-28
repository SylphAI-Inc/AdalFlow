"""Base class for Autograd Components that can be called and backpropagated through."""

from typing import TYPE_CHECKING, Callable, Optional
from collections import OrderedDict
import uuid
import logging
from copy import deepcopy

if TYPE_CHECKING:
    from adalflow.core.generator import BackwardEngine
    from adalflow.optim.parameter import Parameter

from adalflow.optim.types import ParameterType

from adalflow.core.component import Component
from adalflow.optim.function import BackwardContext
from adalflow.utils.registry import EntityMapping


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

    def __init__(self, *args, **kwargs):
        super().__init__()
        super().__setattr__("backward_engine", None)
        super().__setattr__("id", str(uuid.uuid4()))

    def __call__(self, *args, **kwargs):
        if self.training:
            return self.forward(*args, **kwargs)
        else:
            return self.call(*args, **kwargs)

    def set_backward_engine(self, backward_engine: "BackwardEngine", *args, **kwargs):
        raise NotImplementedError("set_backward_engine method is not implemented")

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

        # if "id" not in kwargs:
        #     raise ValueError(
        #         "id must be provided in the kwargs of a GradComponent for tracing."
        #     )

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
        for v in kwargs.values():
            if isinstance(v, Parameter):
                predecessors.append(v)
                if v.param_type == ParameterType.INPUT:
                    v.data_id = kwargs.get("id", None)
                if data_id is None:
                    data_id = v.data_id

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
        # input_args.update(kwargs)
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
            )
        )
        return response

    def backward(self, *, response: "Parameter", id: str = None, **kwargs):
        """Backward pass of the function. In default, it will pass all the scores to the predecessors.

        Note: backward is mainly used internally and better to only allow kwargs as the input.

        Subclass should implement this method if you need additional backward logic.
        """
        from adalflow.optim.parameter import GradientContext

        log.info(f"GradComponent backward: {response.name}")
        children_params = response.predecessors

        if response.get_gradient_and_context_text().strip() == "":
            log.info(f"Generator: Backward: No gradient found for {response}.")

        # backward the backward engine disable signal
        if response.backward_engine_disabled:
            for pred in children_params:
                pred.backward_engine_disabled = True

        for _, pred in enumerate(children_params):
            pred.set_score(response._score)

            if pred.param_type == ParameterType.DEMOS:
                pred.add_score_to_trace(
                    trace_id=id, score=response._score, is_teacher=self.teacher_mode
                )

            # pass the current gradient to pred

            # TODO: each gradcomponent will have its own context, but
            # passing the successor's gradient.data to the current.

            for grad in response.gradients:
                # NOTE: make a copy of the gradient, we should not modify the original gradient
                grad = deepcopy(grad)
                # update the gradient context and from and to
                # grad.update_from_to(response, pred)
                grad.is_default_copy = (
                    True  # response and pred will keep the original gradient
                )
                grad.add_context(
                    GradientContext(
                        variable_desc=pred.role_desc,
                        response_desc=response.name,
                        input_output=f"""{response.component_trace.to_context_str()}""",
                    )
                )

                pred.add_gradient(grad)


class FunGradComponent(GradComponent):
    r"""Wraps a function as a GradComponent.

    Args:
        fun (Callable): The function to be wrapped.

    Examples:

    function = lambda x: x + 1
    fun_component = FunComponent(function)
    print(fun_component(1))  # 2
    """

    def __init__(self, fun: Optional[Callable] = None, afun: Optional[Callable] = None):
        super().__init__()
        self.fun_name = fun.__name__
        EntityMapping.register(self.fun_name, fun)

    def call(self, *args, **kwargs):
        fun = EntityMapping.get(self.fun_name)
        return fun(*args, **kwargs)

    def _extra_repr(self) -> str:
        return super()._extra_repr() + f"fun_name={self.fun_name}"


def fun_to_grad_component(fun) -> FunGradComponent:
    r"""Helper function to convert a function into a Component with
    its own class name.

    Can be used as both a decorator and a function.

    Args:
        fun (Callable): The function to be wrapped.
    Returns:
        FunComponent: The component that wraps the function.

    Examples:
    1. As a decorator:
        >>> @fun_to_component
        >>> def my_function(x):
        >>>     return x + 1
        >>> # is equivalent to
        >>> class MyFunctionComponent(FunComponent):
        >>>     def __init__(self):
        >>>         super().__init__(my_function)

    2. As a function:
        >>> my_function_component = fun_to_component(my_function)
    """

    # Split the function name by underscores, capitalize each part, and join them back together
    class_name = (
        "".join(part.capitalize() for part in fun.__name__.split("_")) + "GradComponent"
    )
    # register the function
    EntityMapping.register(fun.__name__, fun)
    # Define a new component class dynamically
    component_class = type(
        class_name,
        (FunGradComponent,),
        {"__init__": lambda self: FunGradComponent.__init__(self, fun)},
    )
    # register the component
    EntityMapping.register(class_name, component_class)

    return component_class()


if __name__ == "__main__":
    # Test FunGradComponent
    from adalflow.optim.parameter import Parameter

    def my_function(x):
        return x + 1

    my_function_component = fun_to_grad_component(my_function)
    print(my_function_component)  # 2
    # eval mode
    output = my_function_component(1)
    print(output)
    # training mode
    my_function_component.train()
    output = my_function_component(Parameter(data=1, name="input"))
    print(output)

    # now test the decorator
    @fun_to_grad_component
    def my_function(x):
        return x + 1

    print(my_function(1))
    # eval mode
    output = my_function(1)
    print(output)
    assert output == 2

    # training mode
    my_function.train()
    output = my_function(Parameter(data=1, name="input"))
    print(output)
