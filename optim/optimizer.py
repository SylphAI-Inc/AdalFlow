from typing import List, Optional
from copy import deepcopy

from core.parameter import Parameter
from core.component import Component
from optim.sampler import Sampler, Sample


class Optimizer:
    def state_dict(self):
        pass

    def step(self, *args, **kwargs):
        raise NotImplementedError("step method is not implemented")


class BootstrapFewShot(Optimizer):
    __doc__ = r"""BootstrapFewShot performs few-shot sampling used in few-shot ICL.

    It simply orchestrates a sampler and an output processor to generate examples."""

    def __init__(
        self,
        parameter: Parameter,
        sampler: Sampler,
        output_processors: Component,
        num_shots: int,
    ):
        super().__init__()
        self.example_parameter = parameter
        self.sampler = sampler
        self.current: List[Sample] = []  # buffer to store the examples
        self.proposed: List[Sample] = []
        self.output_processors = output_processors
        self.num_shots = num_shots
        # self.proposing = False

    def init(self, weights: Optional[List[float]] = None):
        r"""Initialize the parameters with the initial examples."""
        self.current = self.sampler(self.num_shots, replace=False)
        self.proposed = deepcopy(self.current)
        examples = deepcopy(self.current)
        if self.output_processors:
            examples = self.output_processors(examples)
        self.example_parameter.update_value(examples)

    def random_replace(
        self, shots: int, weights_per_class: Optional[List[float]] = None
    ):
        print(f"before random_replace: {self.current}")
        assert (
            len(self.current) == self.num_shots
        ), f"Ensure you have called init() first to setup the current examples before replacing a subset of them."
        self.proposed = self.sampler.random_replace(
            shots, deepcopy(self.current), weights_per_class=weights_per_class
        )
        print(f"after random_replace: {self.proposed}")

    def propose(self, shots: int, weights_per_class: Optional[List[float]] = None):
        r"""
        Update parameter with the proposed examples.
        """
        # TODO: the replaced shots as the step goes should decrease as it converges
        self.random_replace(shots=shots, weights_per_class=weights_per_class)
        examples = deepcopy(self.proposed)
        if self.output_processors:
            examples = self.output_processors(examples)
        self.example_parameter.update_value(examples)
        # self.proposing = True
        return examples

    def update_parameter(self):
        """Load the proposed into the current."""
        self.current = deepcopy(self.proposed)
        self.proposed = []
        # self.proposing = False
        examples = deepcopy(self.current)
        if self.output_processors:
            examples = self.output_processors(examples)
        self.example_parameter.update_value(examples)

    def reset_parameter(self):
        r"""When performance did not improve, reset the parameter to the current examples."""
        examples = deepcopy(self.current)
        if self.output_processors:
            examples = self.output_processors(examples)
        self.example_parameter.update_value(examples)
        self.proposed = []
        # self.proposing = False
