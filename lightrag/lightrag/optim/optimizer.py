from typing_extensions import TypeAlias
from typing import Dict, Any, Union, Iterable

from lightrag.optim.parameter import Parameter

ParamsT: TypeAlias = Union[Iterable[Parameter], Iterable[Dict[str, Any]]]


class Optimizer:
    r"""Base class for all optimizers."""

    # def __init__(self, params: ParamsT):
    #     r"""Initialize the optimizer.

    #     Args:
    #         params: The parameters to optimize.
    #     """
    #     self.params = params

    # Allow both a list of parameters and a single parameters
    # ORPO optimize a single parameter
    # @overload
    # def __init__(self, params: ParamsT):
    #     r"""Initialize the optimizer."""
    #     ...

    # @overload
    # def __init__(self, parameter: Parameter):
    #     r"""Initialize the optimizer."""
    #     ...

    # def __init__(self, params: Union[ParamsT, Parameter]):
    #     r"""Initialize the optimizer."""
    #     if isinstance(params, Parameter):
    #         self.params = [params]
    #     else:
    #         self.params = params
    proposing: bool = False
    params: ParamsT

    def state_dict(self):
        pass

    def step(self, *args, **kwargs):
        raise NotImplementedError("step method is not implemented")

    def zero_grad(self):
        """Clear all the gradients of the parameters."""
        raise NotImplementedError("zero_grad method is not implemented")
