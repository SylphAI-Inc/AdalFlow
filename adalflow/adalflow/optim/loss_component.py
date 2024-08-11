"""Base class for Autograd Components that can be called and backpropagated through."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adalflow.core.generator import BackwardEngine
    from adalflow.optim.parameter import Parameter

from adalflow.core.component import Component


class LossComponent(Component):
    __doc__ = """A base class to define interfaces for an auto-grad component/operator.

    Compared with `Component`, `GradComponent` defines three important interfaces:
    - `forward`: the forward pass of the function, returns a `Parameter` object that can be traced and backpropagated.
    - `backward`: the backward pass of the function, updates the gradients/prediction score backpropagated from a "loss" parameter.
    - `set_backward_engine`: set the backward engine(a form of generator) to the component, which is used to backpropagate the gradients.

    The __call__ method will check if the component is in training mode,
    and call the `forward` method to return a `Parameter` object if it is in training mode,
    otherwise, it will call the `call` method to return the output such as "GeneratorOutput", "RetrieverOutput", etc.
    """
    backward_engine: "BackwardEngine"
    _component_type = "loss"

    def __init__(self, *args, **kwargs):
        super().__init__()
        super().__setattr__("backward_engine", None)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def set_backward_engine(self, backward_engine: "BackwardEngine", *args, **kwargs):
        raise NotImplementedError("set_backward_engine method is not implemented")

    def forward(self, *args, **kwargs) -> "Parameter":
        r"""Default just wraps the call method."""
        raise NotImplementedError("forward method is not implemented")

    def backward(self, *args, **kwargs):
        raise NotImplementedError("backward method is not implemented")
