from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generator import BackwardEngine
    from lightrag.optim.parameter import Parameter

from .component import Component


class GradComponent(Component):
    __doc__ = """The class to define a function that can be called and backpropagated through."""
    backward_engine: "BackwardEngine"

    def __init__(self, *args, **kwargs):
        # super().__init__()
        super().__init__()
        super().__setattr__("backward_engine", None)

    def __call__(self, *args, **kwargs):
        if self.training:
            return self.forward(*args, **kwargs)
        else:
            return self.call(*args, **kwargs)

    def set_backward_engine(self, backward_engine: "BackwardEngine", *args, **kwargs):
        raise NotImplementedError("set_backward_engine method is not implemented")

    def forward(self, *args, **kwargs) -> "Parameter":
        r"""Default just wraps the call method."""
        # from lightrag.optim.parameter import Parameter

        # data = self.call(*args, **kwargs)
        # output: Parameter = Parameter(data=data, alias=f"{self.name}_output")
        # return output
        raise NotImplementedError("forward method is not implemented")

    def backward(self, *args, **kwargs):
        raise NotImplementedError("backward method is not implemented")
