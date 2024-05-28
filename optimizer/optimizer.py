from typing import Mapping


from core.parameter import Parameter
from core.component import Component


class Optimizer:
    def state_dict(self):
        pass


r"""
We focus on error fixing, run a batch, get batch based accuracy.

# pass the batch to the LLMOptimizer

# sample a class from the batch.Let an llm to boostra
"""


class BootstrapFewShot(Optimizer):
    __doc__ = r"""The optimizer simply performs few-shot sampling used in few-shot ICL.

    It simply orchestrates a sampler and an output processor to generate examples."""

    def __init__(
        self,
        parameter_dict: Mapping[str, Parameter],
        parameter_name: str,
        sampler: Component,
        output_processors: Component,
    ):
        super().__init__()
        self.example_parameter = parameter_dict[parameter_name]
        self.parameter_dict = parameter_dict
        self.sampler = sampler
        self.output_processors = output_processors

    def step(self, num_shots: int):
        examples = self.sampler(num_shots)
        if self.output_processors:
            examples = self.output_processors(examples)
        self.example_parameter.update_value(examples)
        return self.example_parameter

    def state_dict(self):
        # TODO: need to figure out how really parameters and states are saved and loaded.
        return {"example_parameter": self.example_parameter}
