from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:
    from lightrag.core.generator import BackwardEngine


# class GradFunction(Component):
#     __doc__ = """The class to define a function that can be called and backpropagated through."""
#     backward_engine: "BackwardEngine"

#     def __init__(self, *args, **kwargs):
#         # super().__init__()
#         super().__init__()
#         super().__setattr__("backward_engine", None)

#     def __call__(self, *args, **kwargs):
#         if self.training:
#             return self.forward(*args, **kwargs)
#         else:
#             return self.call(*args, **kwargs)

#     def set_backward_engine(self, backward_engine: "BackwardEngine", *args, **kwargs):
#         raise NotImplementedError("set_backward_engine method is not implemented")

#     def forward(self, *args, **kwargs) -> "Parameter":
#         r"""Default just wraps the call method."""
#         # from lightrag.optim.parameter import Parameter

#         # data = self.call(*args, **kwargs)
#         # output: Parameter = Parameter(data=data, alias=f"{self.name}_output")
#         # return output
#         raise NotImplementedError("forward method is not implemented")

#     def backward(self, *args, **kwargs):
#         raise NotImplementedError("backward method is not implemented")


class BackwardContext:
    """
    Represents a context for backward computation.

    :param backward_fn: The backward function to be called during backward computation.
    :type backward_fn: callable
    :param args: Variable length argument list to be passed to the backward function.
    :param kwargs: Arbitrary keyword arguments to be passed to the backward function.

    :ivar backward_fn: The backward function to be called during backward computation.
    :vartype backward_fn: callable
    :ivar fn_name: The fully qualified name of the backward function.
    :vartype fn_name: str
    :ivar args: Variable length argument list to be passed to the backward function.
    :ivar kwargs: Arbitrary keyword arguments to be passed to the backward function.

    :method __call__(backward_engine: EngineLM) -> Any:
        Calls the backward function with the given backward engine and returns the result.
    :method __repr__() -> str:
        Returns a string representation of the BackwardContext object.
    """

    def __init__(
        self,
        backward_fn: Callable,
        backward_engine: "BackwardEngine" = None,
        *args,
        **kwargs,
    ):
        self.backward_fn = backward_fn
        self.backward_engine = backward_engine
        self.fn_name = f"{backward_fn.__module__}.{backward_fn.__qualname__}"
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        if self.backward_engine is None:
            return self.backward_fn(*self.args, **self.kwargs)

        return self.backward_fn(
            *self.args, **self.kwargs, backward_engine=self.backward_engine
        )

    def __repr__(self):
        return f"{self.fn_name}"
