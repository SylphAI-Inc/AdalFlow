from typing_extensions import TypeAlias
from typing import Dict, Any, Union, Iterable, Sequence

from lightrag.optim.parameter import Parameter
from lightrag.optim.sampler import Sampler
from lightrag.core.base_data_class import DataClass

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

    def propose(self, *args, **kwargs):
        raise NotImplementedError("propose method is not implemented")

    def step(self, *args, **kwargs):
        raise NotImplementedError("step method is not implemented")

    def revert(self, *args, **kwargs):
        raise NotImplementedError("revert method is not implemented")


class TextOptimizer(Optimizer):
    __doc__ = r"""Base class for all text optimizers.

    Text optimizer is via textual gradient descent, which is a variant of gradient descent that optimizes the text directly.
    It will generate new values for a given text prompt.This includes:
    - System prompt
    - output format
    - prompt template
    """

    def __init__(self, *args, **kwargs):
        pass

    def zero_grad(self):
        """Clear all the gradients of the parameters."""
        raise NotImplementedError("zero_grad method is not implemented")


class DemoOptimizer(Optimizer):

    __doc__ = r"""Base class for all demo optimizers.

    Demo optimizer are few-shot optimization, where it will sample raw examples from train dataset or bootstrap examples from the model's output.
    It will work with a sampler to generate new values for a given text prompt.

    If bootstrap is used, it will require a teacher genearator to generate the examples.
    """

    sampler: Sampler
    _traces: Dict[str, Any]  # key: parameter_id (demo)
    dataset: Sequence[DataClass]

    def __init__(self, *args, **kwargs):
        pass

    def init_shots(self, *args, **kwargs):
        r"""Initialize the samples for each parameter."""
        raise NotImplementedError("init method is not implemented")

    def set_dataset(self, dataset: Sequence[DataClass]):
        r"""Set the dataset for the optimizer."""
        self.dataset = dataset
