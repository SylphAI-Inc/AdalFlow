"""Adapted and optimized boostrap fewshot optimizer:

DSPy: Com-piling declarative language model calls into state-of-the-art pipelines."""

from typing import List, Optional, Dict
import logging

from adalflow.core.base_data_class import DataClass


from adalflow.optim.parameter import Parameter


from adalflow.core.functional import random_sample
from adalflow.optim.optimizer import DemoOptimizer
from adalflow.optim.types import ParameterType

log = logging.getLogger(__name__)


class BootstrapFewShot(DemoOptimizer):
    __doc__ = r"""BootstrapFewShot performs few-shot sampling used in few-shot ICL.

    It will be used to optimize paramters of demos.
    Based on research from AdalFlow team and DsPy library.

    Compared with Dspy's version:
     1. we added weighted sampling for both the raw and augmented demos
        to prioritize failed demos but successful in augmented demos based on the evaluation score
        while we backpropagate the demo samples.
     2. In default, we exclude the input fields from the augmented demos. Our reserch finds that
        using the reasoning demostrations from teacher model can be more effective in some cases than taking both inputs and output
        samples and be more token efficient.

    Reference:
    - DsPy: Com-piling declarative language model calls into state-of-the-art pipelines.
    """
    exclude_input_fields_from_bootstrap_demos: bool

    def __init__(
        self,
        params: List[Parameter],
        raw_shots: Optional[int] = None,
        bootstrap_shots: Optional[int] = None,
        dataset: Optional[List[DataClass]] = None,
        weighted: bool = True,
        exclude_input_fields_from_bootstrap_demos: bool = False,
    ):
        super().__init__(weighted=weighted, dataset=dataset)
        self.params = [
            param
            for param in params
            if param.requires_opt and param.param_type == ParameterType.DEMOS
        ]
        log.info(f"BootstrapFewShot: {self.params}")

        self._raw_shots = raw_shots
        self._bootstrap_shots = bootstrap_shots

        self.proposing = False
        self._teacher_scores: Dict[str, float] = {}  # data id to score
        self._student_scores: Dict[str, float] = {}  # data id to score
        self.exclude_input_fields_from_bootstrap_demos = (
            exclude_input_fields_from_bootstrap_demos
        )

    def add_scores(self, ids: List[str], scores: List[float], is_teacher: bool = True):
        if len(ids) != len(scores):
            raise ValueError("ids and scores must have the same length")

        for score in scores:

            if not isinstance(score, float):
                raise ValueError(
                    f"score must be a float, got {type(score)}, score: {score}"
                )
            if score < 0 or score > 1:
                raise ValueError("score must be in range [0, 1]")

        target = self._teacher_scores if is_teacher else self._student_scores
        for i, id in enumerate(ids):
            target[id] = scores[i]

    def config_shots(self, raw_shots: int, bootstrap_shots: int):
        self._raw_shots = raw_shots
        self._bootstrap_shots = bootstrap_shots

    def config_dataset(self, dataset: List[DataClass]):
        self.dataset = dataset

    def _pre_check(self):
        if not self.dataset:
            raise ValueError("dataset must be provided")
        if self._raw_shots is None or self._bootstrap_shots is None:
            raise ValueError("raw_shots and bootstrap_shots must be provided")

    @property
    def num_shots(self) -> int:
        return (
            self._raw_shots + self._bootstrap_shots
            if self._raw_shots and self._bootstrap_shots
            else 0
        )

    def sample(
        self,
        augmented_demos: Dict[str, DataClass],
        demos: Dict[str, DataClass],
        dataset: List[DataClass],
        raw_shots: int,
        bootstrap_shots: int,
        weighted: bool = True,
    ):
        r"""Performs weighted sampling, ensure the score is in range [0, 1]. The higher score means better accuracy."""
        # 1. sample from augmented demos
        # set weights to be score
        # add 1 to all score to avoid negative weights
        augmented_options = list(augmented_demos.values())
        weights = None
        if weighted:
            weights: List[float] = []
            for demo in augmented_options:
                demo_score = self._teacher_scores.get(demo.id, None)
                if demo_score is None:
                    raise ValueError(
                        f"score must be provided for each demo, id: {demo.id}, all scores: {self._teacher_scores}"
                    )
                if demo_score < 0 or demo_score > 1:
                    raise ValueError(f"score must be in range [0, 1], got {demo_score}")

                w = demo_score
                student_demo_score = self._student_scores.get(demo.id, None)

                if student_demo_score is not None:
                    if student_demo_score < 0 or student_demo_score > 1:
                        raise ValueError(
                            f"score must be in range [0, 1], got {student_demo_score}"
                        )
                    # if demo.id in demos and demos[demo.id].score is not None:
                    w = (
                        w
                        - student_demo_score
                        # w - demos[demo.id].score
                    )  # assign higher weights to failed demos but successful in augmented
                    if w < 0:
                        w = 0
                weights.append(w)

        # print(f"augs: {augmented_options}")
        sampled_augmented_demos = (
            random_sample(
                augmented_options, bootstrap_shots, replace=False, weights=weights
            )
            if len(augmented_options) > 0
            else []
        )

        # 2. sample from raw demos
        # exclude the sampled augmented demos
        # TODO: ensure all data points has unique ids
        filtered_dataset = list(
            filter(
                lambda x: x.id
                not in set([demo.id for demo in sampled_augmented_demos]),
                dataset,
            )
        )
        if len(filtered_dataset) == 0:
            # If no demos left we will get raw_weights [], sum to 0
            return sampled_augmented_demos, []
        # assigne weights 0 to all options
        raw_weights = None
        if weighted:
            raw_weights = [0.0] * len(filtered_dataset)
            # for those exist in the demos, assign higher score with failed demos
            for i, demo in enumerate(filtered_dataset):
                student_demo_score = self._student_scores.get(demo.id, None)

                if student_demo_score is not None:
                    # ensure the score is in range [0, 1]
                    if student_demo_score < 0 or student_demo_score > 1:
                        raise ValueError(
                            f"score must be in range [0, 1], got {student_demo_score}"
                        )
                    raw_weights[i] = 1 - student_demo_score

        sampled_raw_demos = random_sample(
            filtered_dataset, raw_shots, replace=False, weights=raw_weights
        )
        return sampled_augmented_demos, sampled_raw_demos

    @staticmethod
    def samples_to_str(
        samples: List[DataClass], augmented: bool = False, exclude_inputs: bool = False
    ) -> str:
        sample_strs = []
        for sample in samples:
            try:

                # process the input fields
                if augmented:
                    exclude_fields = ["id", "score"]
                    if exclude_inputs:
                        exclude_fields.extend(sample.get_input_fields())
                    yaml_str = sample.to_yaml(exclude=exclude_fields)

                else:
                    yaml_str = sample.to_yaml(exclude=["id", "score"])
                sample_strs.append(yaml_str + "\n")
            except Exception as e:
                print(f"Error: {e} to yaml for {sample}")
                sample_strs.append(str(sample))
        return "\n".join(sample_strs)

    def propose(self):
        r"""Proposing a value while keeping previous value saved on parameter."""
        self._pre_check()
        if self.proposing:
            raise ValueError("Already proposing a value.")
        for demo_param in self.params:
            if demo_param.requires_opt:
                augmented_demos = demo_param._traces
                demos = demo_param._student_traces
                try:
                    sampled_augmented_demos, sampled_raw_demos = self.sample(
                        augmented_demos=augmented_demos,
                        demos=demos,
                        dataset=self.dataset,
                        raw_shots=self._raw_shots,
                        bootstrap_shots=self._bootstrap_shots,
                        weighted=self._weighted,
                    )
                    print(
                        f"sampled_augmented_demos: {[demo.id for demo in sampled_augmented_demos]}"
                    )
                    samples = sampled_augmented_demos + sampled_raw_demos

                    demo_str = ""
                    if len(sampled_augmented_demos) > 0:

                        demo_str = self.samples_to_str(
                            samples=sampled_augmented_demos,
                            augmented=True,
                            exclude_inputs=self.exclude_input_fields_from_bootstrap_demos,
                        )
                    if len(sampled_raw_demos) > 0:
                        demo_str += "\n" + self.samples_to_str(
                            samples=sampled_raw_demos, augmented=False
                        )
                    demo_str = demo_str.strip()
                    demo_param.propose_data(demo_str, samples)
                except Exception as e:
                    print(f"Error: {e} for {demo_param.name}")
                    raise e

        self.proposing = True

    def revert(self):
        """Revert to the previous value when the evaluation is worse."""
        self._pre_check()
        if not self.proposing:
            raise ValueError("Not proposing a value.")
        for param in self.params:
            param.revert_data(include_demos=True)
        self.proposing = False

    def step(self):
        """Discard the previous value and keep the proposed value."""
        self._pre_check()
        if not self.proposing:
            raise ValueError("Not proposing a value.")
        for param in self.params:
            param.step_data(include_demos=True)
            # TODO: track all past history

        self.proposing = False

    def __str__(self) -> str:
        s = f"BootstrapFewShot(raw_shots={self._raw_shots}, bootstrap_shots={self._bootstrap_shots}, \
            params={[p.name for p in self.params]}, dataset={len(self.dataset) if self.dataset else 0})"
        return s

    def __repr__(self) -> str:
        return self.__str__()
