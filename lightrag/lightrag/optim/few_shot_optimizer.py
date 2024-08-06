from typing import List, Optional, Dict, TYPE_CHECKING
from copy import deepcopy
import logging

from lightrag.core.base_data_class import DataClass


from lightrag.optim.parameter import Parameter

if TYPE_CHECKING:
    from lightrag.core.component import Component
from lightrag.optim.sampler import Sampler, Sample, random_sample
from lightrag.optim.optimizer import DemoOptimizer
from lightrag.optim.types import FewShotConfig, ParameterType

log = logging.getLogger(__name__)


class BootstrapFewShot(DemoOptimizer):
    __doc__ = r"""BootstrapFewShot performs few-shot sampling used in few-shot ICL.

    It simply orchestrates a sampler and an output processor to generate examples."""

    def __init__(
        self,
        params: List[Parameter],
        few_shot_config: FewShotConfig,
        # num_shots: int,
        # num_raw_demos: int,
        # num_augmented_demos: int,
        # llm_augmenter: Optional["Component"] = None,
        sampler: Optional[Sampler] = None,
        task_input_dataclass: Optional[DataClass] = None,
        output_processors: Optional["Component"] = None,
        task_output_dataclass: Optional[DataClass] = None,
    ):
        super().__init__()
        self.params = [
            param
            for param in params
            if param.requires_opt and param.param_type == ParameterType.DEMOS
        ]
        log.info(f"BootstrapFewShot: {self.params}")
        self.sampler: Sampler = sampler
        self.current: List[Sample] = []  # buffer to store the examples
        self.proposed: List[Sample] = []
        self.output_processors = output_processors
        self.few_shot_config = few_shot_config
        if (
            self.few_shot_config.raw_shots + self.few_shot_config.bootstrap_shots
            != self.few_shot_config.num_shots
        ):
            raise ValueError("raw_shots + bootstrap_shots must equal num_shots")

        self._num_shots = self.few_shot_config.num_shots
        self._raw_shots = self.few_shot_config.raw_shots
        self._bootstrap_shots = self.few_shot_config.bootstrap_shots
        # self.llm_augmenter = llm_augmenter
        self.task_input_dataclass = task_input_dataclass
        self.task_output_dataclass = task_output_dataclass
        # if llm_augmenter is not None:
        #     if task_input_dataclass is None or task_output_dataclass is None:
        #         raise ValueError(
        #             "task_input_dataclass and task_output_dataclass must be provided when llm_augment is not None"
        #         )
        # self.proposing = False

    def set_sampler(self, sampler: Sampler):
        self.sampler = sampler

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

    def init_shots(self, shots: Optional[int] = None):
        r"""Initialize the parameters with the initial examples."""
        if shots is None:
            shots = self._num_shots
        if self.sampler is None:
            raise ValueError("sampler must be provided")
        if self.sampler.dataset is None:
            raise ValueError(
                "sampler dataset must be provided to initialize the samples"
            )
        self.current = self.sampler(
            shots, replace=False
        )  # this is the end to end dataset
        print(f"current: {self.current}")
        # if self.llm_augmenter:
        #     self.current = self.augment_samples(self.current)
        self.proposed = deepcopy(self.current)
        examples = deepcopy(self.current)
        if self.output_processors:
            examples = self.output_processors(examples)
        for param in self.params:
            param.update_value(examples)
            print(f"param: {param}")

    def random_replace(
        self, shots: int, weights_per_class: Optional[List[float]] = None
    ):
        assert (
            len(self.current) == self.num_shots
        ), "Ensure you have called init() first to setup the current examples before replacing a subset of them."
        self.proposed = self.sampler.random_replace(
            shots, deepcopy(self.current), weights_per_class=weights_per_class
        )

    def propose(self):
        for demo_param in self.params:
            if demo_param.requires_opt:
                # sample augmented demos
                # TODO: teacher mode should only have demo backpropogation
                augmented_demos = demo_param._traces
                print(f"augmented_demos: {augmented_demos}")
                options = list(augmented_demos.values())
                filtered_options = list(filter(lambda x: x.score > 0.5, options))

                sampled_demos = random_sample(
                    filtered_options, self._bootstrap_shots, replace=False
                )
                demo_strs = []
                for demo in sampled_demos:
                    demo_strs.append(demo.to_yaml(exclude=["id", "score"]))
                demo_str = "\n".join(demo_strs)
                # sample raw demos

                demo_param.propose_data(demo_str)
                print(f"demo_param value: {demo_param.data}")

    def propose_old(self, shots: int, weights_per_class: Optional[List[float]] = None):
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

    def step(self):
        """Load the proposed into the current."""
        self.current = deepcopy(self.proposed)
        self.proposed = []
        # self.proposing = False
        examples = deepcopy(self.current)
        if self.output_processors:
            examples = self.output_processors(examples)
        self.example_parameter.update_value(examples)

    def revert(self):
        r"""When performance did not improve, reset the parameter to the current examples."""
        examples = deepcopy(self.current)
        if self.output_processors:
            examples = self.output_processors(examples)
        self.example_parameter.update_value(examples)
        self.proposed = []
        # self.proposing = False
