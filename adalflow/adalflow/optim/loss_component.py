"""Base class for Autograd Components that can be called and backpropagated through."""

from typing import TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from adalflow.core.generator import BackwardEngine
    from adalflow.optim.parameter import Parameter

from adalflow.core.component import Component


# TODO: make it a subclass of GradComponent
class LossComponent(Component):
    __doc__ = """A base class to define a loss component.

    Loss component is to compute the textual gradients/feedback for each of its predecessors using another LLM as the backward engine.

    Each precessor should have basic information that is passed to its next component to inform its type such as retriever or generator and its role description.

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
    id = None
    _disable_backward_engine: bool

    def __init__(self, *args, **kwargs):
        super().__init__()
        super().__setattr__("backward_engine", None)
        super().__setattr__("id", str(uuid.uuid4()))
        super().__setattr__("_disable_backward_engine", False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def set_backward_engine(self, backward_engine: "BackwardEngine", *args, **kwargs):
        raise NotImplementedError("set_backward_engine method is not implemented")

    def disable_backward_engine(self):
        r"""Does not run gradients generation, but still with backward to gain module-context"""
        self._disable_backward_engine = True

    def forward(self, *args, **kwargs) -> "Parameter":
        r"""Default just wraps the call method."""
        raise NotImplementedError("forward method is not implemented")

    def backward(self, *args, **kwargs):
        raise NotImplementedError("backward method is not implemented")
