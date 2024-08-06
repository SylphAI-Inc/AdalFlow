from typing import List, Optional, Dict
import logging

from lightrag.core.base_data_class import DataClass


from lightrag.optim.parameter import Parameter


from lightrag.core.functional import random_sample
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
        dataset: Optional[List[DataClass]] = None,
    ):
        super().__init__()
        self.params = [
            param
            for param in params
            if param.requires_opt and param.param_type == ParameterType.DEMOS
        ]
        log.info(f"BootstrapFewShot: {self.params}")

        # self.output_processors = output_processors
        self.few_shot_config = few_shot_config
        if (
            self.few_shot_config.raw_shots + self.few_shot_config.bootstrap_shots
            != self.few_shot_config.num_shots
        ):
            raise ValueError("raw_shots + bootstrap_shots must equal num_shots")

        self._num_shots = self.few_shot_config.num_shots
        self._raw_shots = self.few_shot_config.raw_shots
        self._bootstrap_shots = self.few_shot_config.bootstrap_shots
        self.proposing = False
        self.dataset = dataset

    # TODO: ensure all demo data structure must have id and score.
    @staticmethod
    def sample(
        augmented_demos: Dict[str, DataClass],
        demos: Dict[str, DataClass],
        dataset: List[DataClass],
        raw_shots: int,
        bootstrap_shots: int,
    ):
        r"""Performs weighted sampling, ensure the score is in range [0, 1]. The higher score means better accuracy."""
        # 1. sample from augmented demos
        # set weights to be score
        augmented_options = list(augmented_demos.values())
        weights: List[float] = []
        for demo in augmented_options:
            if demo.score is None:
                raise ValueError("score must be provided for each demo")
            w = demo.score
            if demo.id in demos and demos[demo.id].score is not None:
                w = (
                    w - demos[demo.id].score
                )  # assign higher weights to failed demos but successful in augmented
            weights.append(w)
        sampled_augmented_demos = random_sample(
            augmented_options, bootstrap_shots, replace=False, weights=weights
        )

        # 2. sample from raw demos
        # exclude the sampled augmented demos
        filtered_dataset = list(
            filter(
                lambda x: x.id
                not in set([demo.id for demo in sampled_augmented_demos]),
                dataset,
            )
        )
        # assigne weights 0 to all options
        raw_weights = [0.0] * len(filtered_dataset)
        # for those exist in the demos, assign higher score with failed demos
        for i, demo in enumerate(filtered_dataset):
            if demo.id in demos and demos[demo.id].score is not None:
                raw_weights[i] = 1 - demos[demo.id].score
        sampled_raw_demos = random_sample(
            filtered_dataset, raw_shots, replace=False, weights=raw_weights
        )
        return sampled_augmented_demos, sampled_raw_demos

    @staticmethod
    def samples_to_str(samples: List[DataClass]) -> str:
        sample_strs = []
        for sample in samples:
            sample_strs.append(sample.to_yaml(exclude=["id", "score"]))
        return "\n".join(sample_strs)

    def propose(self):
        r"""Proposing a value while keeping previous value saved on parameter."""
        if self.proposing:
            raise ValueError("Already proposing a value.")
        for demo_param in self.params:
            if demo_param.requires_opt:
                augmented_demos = demo_param._traces
                demos = demo_param._student_traces
                sampled_augmented_demos, sampled_raw_demos = self.sample(
                    augmented_demos=augmented_demos,
                    demos=demos,
                    dataset=self.dataset,
                    raw_shots=self._raw_shots,
                    bootstrap_shots=self._bootstrap_shots,
                )
                samples = sampled_augmented_demos + sampled_raw_demos
                demo_str = self.samples_to_str(samples=samples)

                demo_param.propose_data(demo_str, samples)
                print(f"demo_param value: {demo_param.data}")
                print(f"demo_param demo: {demo_param._demos}")
        self.proposing = True

    def revert(self):
        """Revert to the previous value when the evaluation is worse."""
        if not self.proposing:
            raise ValueError("Not proposing a value.")
        for param in self.params:
            param.revert_data(include_demos=True)
        self.proposing = False

    def step(self):
        """Discard the previous value and keep the proposed value."""
        if not self.proposing:
            raise ValueError("Not proposing a value.")
        for param in self.params:
            param.step_data(include_demos=True)
            # TODO: track all past history

        self.proposing = False
