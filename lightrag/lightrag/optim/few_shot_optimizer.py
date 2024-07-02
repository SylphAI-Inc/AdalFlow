from typing import List, Optional, Dict
from copy import deepcopy

from lightrag.core.base_data_class import DataClass
from lightrag.core.parameter import Parameter
from lightrag.core.component import Component
from lightrag.optim.sampler import Sampler, Sample
from lightrag.optim.optimizer import Optimizer


class BootstrapFewShot(Optimizer):
    __doc__ = r"""BootstrapFewShot performs few-shot sampling used in few-shot ICL.

    It simply orchestrates a sampler and an output processor to generate examples."""

    def __init__(
        self,
        parameter: Parameter,
        sampler: Sampler,
        num_shots: int,
        llm_augmenter: Optional[Component] = None,
        task_input_dataclass: Optional[DataClass] = None,
        output_processors: Optional[Component] = None,
        task_output_dataclass: Optional[DataClass] = None,
    ):
        super().__init__()
        self.example_parameter = parameter
        self.sampler = sampler
        self.current: List[Sample] = []  # buffer to store the examples
        self.proposed: List[Sample] = []
        self.output_processors = output_processors
        self.num_shots = num_shots
        self.llm_augmenter = llm_augmenter
        self.task_input_dataclass = task_input_dataclass
        self.task_output_dataclass = task_output_dataclass
        if llm_augmenter is not None:
            if task_input_dataclass is None or task_output_dataclass is None:
                raise ValueError(
                    "task_input_dataclass and task_output_dataclass must be provided when llm_augment is not None"
                )
        # self.proposing = False

    def augment_samples(self, samples: List[Sample]) -> List[Sample]:
        if self.llm_augmenter:  # TODO: better represent sample
            augmented_samples: List[Sample] = []
            for sample in samples:
                sample_data = sample.data
                input_obj = self.task_input_dataclass.from_dict(sample.data)
                output_obj = self.task_output_dataclass.from_dict(sample.data)
                augmented_output_obj: Dict = self.llm_augmenter(input_obj, output_obj)
                # update the fields in the output_obj
                for key, value in augmented_output_obj.items():
                    if hasattr(output_obj, key):
                        # update the field
                        print(f"updating field: {key} with value: {value}")
                        # output_obj.set_field_value(key, value)
                        sample_data[key] = value

                augmented_samples.append(Sample(index=sample.index, data=sample_data))
            # print(f"augmented_samples: {augmented_samples}")
            return augmented_samples

        return samples

    def reset(self):
        self.current = []
        self.proposed = []

    def init(self, weights: Optional[List[float]] = None, shots: Optional[int] = None):
        r"""Initialize the parameters with the initial examples."""
        if shots is None:
            shots = self.num_shots
        self.current = self.sampler(shots, replace=False)
        if self.llm_augmenter:
            self.current = self.augment_samples(self.current)
        self.proposed = deepcopy(self.current)
        examples = deepcopy(self.current)
        if self.output_processors:
            examples = self.output_processors(examples)
        self.example_parameter.update_value(examples)

    def random_replace(
        self, shots: int, weights_per_class: Optional[List[float]] = None
    ):
        assert (
            len(self.current) == self.num_shots
        ), "Ensure you have called init() first to setup the current examples before replacing a subset of them."
        self.proposed = self.sampler.random_replace(
            shots, deepcopy(self.current), weights_per_class=weights_per_class
        )

    def propose(self, shots: int, weights_per_class: Optional[List[float]] = None):
        r"""
        Update parameter with the proposed examples.
        """
        self.random_replace(shots=shots, weights_per_class=weights_per_class)
        if self.llm_augmenter:
            self.proposed = self.augment_samples(self.proposed)
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
