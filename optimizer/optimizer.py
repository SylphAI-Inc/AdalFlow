from copy import deepcopy
from core.parameter import Parameter
from core.component import Component

from optimizer.sampler import RandomSampler


class Optimizer:
    def state_dict(self):
        pass


r"""
We focus on error fixing, run a batch, get batch based accuracy.

# pass the batch to the LLMOptimizer

# sample a class from the batch.Let an llm to boostra
"""


class BootstrapFewShot(Optimizer):
    def __init__(
        self,
        example_parameter: Parameter,
        train_dataset,
        sampler: Component,
        output_processors: Component,
    ):
        super().__init__()
        self.example_parameter = deepcopy(example_parameter)
        self.train_dataset = train_dataset
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
