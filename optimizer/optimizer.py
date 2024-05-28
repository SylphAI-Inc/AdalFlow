from typing import Mapping, Any, Optional, List, Tuple, Callable
import numpy as np
import math
from copy import deepcopy

from core.parameter import Parameter
from core.component import Component
from optimizer.sampler import ClassSampler


class Optimizer:
    def state_dict(self):
        pass

    def step(self, *args, **kwargs):
        raise NotImplementedError("step method is not implemented")


r"""
We focus on error fixing, run a batch, get batch based accuracy.

# pass the batch to the LLMOptimizer

# sample a class from the batch.Let an llm to boostra
"""


class BootstrapFewShot(Optimizer):
    __doc__ = r"""BootstrapFewShot performs few-shot sampling used in few-shot ICL.

    It simply orchestrates a sampler and an output processor to generate examples."""

    def __init__(
        self,
        parameter_dict: Mapping[str, Parameter],
        parameter_name: str,
        # parameter: Parameter,
        sampler: Component,
        output_processors: Component,
        num_shots: int,
    ):
        super().__init__()
        self.example_parameter = parameter_dict[parameter_name]

        # self.parameter_dict = parameter_dict
        self.sampler = sampler
        self.current = []  # buffer to store the examples
        self.proposed = []
        self.output_processors = output_processors
        self.num_shots = num_shots
        self.proposing = False

    def init(self):
        self.current = self.sampler(self.num_shots)
        self.proposed = deepcopy(self.current)
        examples = deepcopy(self.current)
        if self.output_processors:
            examples = self.output_processors(examples)
        self.example_parameter.update_value(examples)

    def random_replace(self, num_shots: int):
        print(f"before random_replace: {self.current}")
        self.proposed = self.sampler.random_replace(num_shots, deepcopy(self.current))
        print(f"after random_replace: {self.proposed}")

    def propose(self, num_shots: int):
        r"""
        Update parameter with the proposed examples.
        """
        self.random_replace(num_shots)  # replace num_shots in the proposed
        examples = deepcopy(self.proposed)
        if self.output_processors:
            examples = self.output_processors(examples)
        self.example_parameter.update_value(examples)
        self.proposing = True
        return examples

    def update_parameter(self):
        """Load the proposed into the current."""
        self.current = deepcopy(self.proposed)
        self.proposed = []
        self.proposing = False
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
        # reset the proposed
        self.proposed = []
        self.proposing = False
