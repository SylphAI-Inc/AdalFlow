from abc import ABC, abstractmethod
from lightrag.optim.parameter import Parameter


class Function(ABC):
    """
    The class to define a function that can be called and backpropagated through.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Parameter:
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass


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

    def __init__(self, backward_fn, backward_engine, *args, **kwargs):
        self.backward_fn = backward_fn
        self.backward_engine = backward_engine
        self.fn_name = f"{backward_fn.__module__}.{backward_fn.__qualname__}"
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.backward_fn(
            *self.args, **self.kwargs, backward_engine=self.backward_engine
        )

    def __repr__(self):
        return f"{self.fn_name}"
