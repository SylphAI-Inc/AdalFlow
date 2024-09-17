"""Base class for Autograd Components that can be called and backpropagated through."""

from typing import TYPE_CHECKING
from collections import OrderedDict
import logging

if TYPE_CHECKING:
    from adalflow.core.generator import BackwardEngine
    from adalflow.optim.parameter import Parameter

from adalflow.optim.types import ParameterType

from adalflow.core.component import Component
from adalflow.optim.function import BackwardContext

__all__ = ["GradComponent"]
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
    """
    backward_engine: "BackwardEngine"
    _component_type = "grad"

    def __init__(self, *args, **kwargs):
        super().__init__()
        super().__setattr__("backward_engine", None)

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

        TODO: all Gradcomponent should not allow args but only kwargs.
        For now, just check if id is in kwargs.
        """

        from adalflow.optim.parameter import Parameter

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

        # Add keyword args to the ordered dict, preserving order
        predecessors = []
        for v in input_args.values():
            if isinstance(v, Parameter):
                predecessors.append(v)
        for v in kwargs.values():
            if isinstance(v, Parameter):
                predecessors.append(v)

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

        # 4. Create a Parameter object to trace the forward pass
        input_args.update(kwargs)
        response = Parameter(
            data=call_response,
            name=self.name + "_output",
            role_desc=self.name + " response",
            param_type=ParameterType.OUTPUT,
        )
        response.set_predecessors(predecessors)
        response.trace_forward_pass(input_args=input_args, full_response=call_response)
        response.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward,
                response=response,
                id=kwargs.get("id", None),
            )
        )
        return response

    def backward(self, *args, **kwargs):
        pass
        # raise NotImplementedError("backward method is not implemented")
