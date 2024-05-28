import random
from dataclasses import dataclass

from typing import List, Sequence, Optional, Callable, Any, Dict
import math

from core.component import Component


@dataclass
class Sample:
    r"""Output data structure for each sampled data in the sequence."""

    index: int  # the initial index of the sample in the dataset
    data: Any  # the data of the sample


class Sampler(Component):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        pass

    def random_replace(self, *args, **kwargs):
        r"""Randomly replace some samples

        Recomnend to have two arguments: shots and samples
        """
        pass

    def call(self, *args, **kwargs) -> Sequence[Sample]:
        r"""Abstract method to do the main sampling"""
        raise NotImplementedError(
            f"call method is not implemented in {type(self).__name__}"
        )


class RandomSampler(Component):
    r"""
    Simple random sampler to sample from the dataset.
    """

    def __init__(self, dataset, default_num_shots: Optional[int] = None):
        super().__init__()
        self.dataset = [Sample(index=i, data=x) for i, x in enumerate(dataset)]

        self.default_num_shots = default_num_shots

    def random_replace(
        self, shots: int, samples: Sequence[Sample], replace: Optional[bool] = False
    ) -> Sequence[Any]:
        r"""
        Randomly replace num of shots in the samples. The replace happens in place.

        If replace is True, it will skip duplicate checks
        """
        assert len(samples) >= shots, "num_samples is larger than the number of samples"
        indices_to_replace = random.sample(range(len(samples)), shots)
        existing_indexces = {sample.index for sample in samples}
        if replace:  # this can potentially result in duplicates in the samples
            for i in indices_to_replace:
                samples[i] = random.choice(self.dataset)
            return samples
        else:
            # exclude the indices in the samples from the choice
            choice_indexces = list(
                set(range(len(self.dataset))) - set(existing_indexces)
            )
            # now sample shots from the choice_indices
            candidates_indices = random.sample(choice_indexces, shots)
            for i, j in zip(indices_to_replace, candidates_indices):
                samples[i] = self.dataset[j]
        return samples

    def random_sample(
        self, shots: int, replace: Optional[bool] = False
    ) -> Sequence[Sample]:
        r"""
        Randomly sample num of shots from the dataset. If replace is True, sample with replacement,
        meaning the same sample can be sampled multiple times.
        """
        if replace:
            return [random.choice(self.dataset) for _ in range(shots)]
        return random.sample(self.dataset, shots)

    def call(
        self, num_shots: Optional[int] = None, replace: Optional[bool] = False
    ) -> Sequence[Sample]:
        if num_shots is None:
            num_shots = self.default_num_shots
        if num_shots is None:
            raise ValueError("num_shots is not set")
        return self.random_sample(num_shots, replace)


class ClassSampler(Component):
    def __init__(
        self,
        dataset,
        num_classes: int,
        get_data_key_fun: Callable,
        default_num_shots: Optional[int] = None,
    ):
        super().__init__()
        self.dataset = [Sample(index=i, data=x) for i, x in enumerate(dataset)]
        self.num_classes = num_classes
        if not get_data_key_fun:
            raise ValueError("get_data_key_fun must be provided")
        self.get_data_key_fun = get_data_key_fun
        self.class_indexces: List[List] = [[] for _ in range(num_classes)]
        for i, data in enumerate(dataset):
            self.class_indexces[self.get_data_key_fun(data)].append(i)

        self.default_num_shots = default_num_shots

    def _sample_one_class(
        self, num_samples: int, class_index: int, replace: Optional[bool] = False
    ) -> Sequence[Sample]:
        r"""
        Sample num_samples from the class with class_index"""
        if replace:
            # TODO: can allow different sample weights to be passed to each class based on the errors
            sampled_indexes = random.choices(
                self.class_indexces[class_index], k=num_samples
            )
        else:
            sampled_indexes = random.sample(
                self.class_indexces[class_index], num_samples
            )
        samples = [self.dataset[i] for i in sampled_indexes]
        return samples

    def random_replace(
        self, shots: int, samples: List[Sample], replace: Optional[bool] = False
    ) -> Sequence[Sample]:
        r"""
        Randomly select num shots from the samples and replace it with another sample has the same class index
        """
        assert len(samples) >= shots, "num_shots is larger than the number of samples"
        existing_indexces_by_class: Dict[Any, List[int]] = {}
        for i, sample in enumerate(samples):
            key = self.get_data_key_fun(sample.data)
            if key not in existing_indexces_by_class:
                existing_indexces_by_class[key] = []
            existing_indexces_by_class[key].append(sample.index)

        # select num shots in samples to replace
        replace_sample_indexes = random.sample(range(len(samples)), shots)
        replace_indexces_by_class: Dict[Any, List[int]] = {}
        for i in replace_sample_indexes:
            key = self.get_data_key_fun(samples[i].data)
            if key not in replace_indexces_by_class:
                replace_indexces_by_class[key] = []
            replace_indexces_by_class[key].append(i)

        # sample for each class and exclude the existing samples
        replace_class_labels = list(replace_indexces_by_class.keys())
        for class_label in replace_class_labels:
            num_sample_per_class = len(replace_indexces_by_class[class_label])
            choice_indexces = list(
                set(self.class_indexces[class_label])
                - set(existing_indexces_by_class[class_label])
            )
            if replace:
                sampled_indexes = random.choices(
                    self.class_indexces[class_label], k=num_sample_per_class
                )

            else:
                sampled_indexes = random.sample(choice_indexces, num_sample_per_class)
            for i, j in zip(replace_indexces_by_class[class_label], sampled_indexes):
                samples[i] = self.dataset[j]

        return samples

    def call(self, num_shots: int, replace: Optional[bool] = False) -> Sequence[Sample]:
        r"""
        Sample num_shots from the dataset. If replace is True, sample with replacement.
        """
        if num_shots is None:
            num_shots = self.default_num_shots
        if num_shots is None:
            raise ValueError("num_shots is not set")

        samples = []
        samples_per_class = math.ceil(num_shots / self.num_classes)
        for class_index in range(self.num_classes):
            samples.extend(
                self._sample_one_class(samples_per_class, class_index, replace)
            )
        if len(samples) > num_shots:
            # randomly sample from the class balance the number of samples
            samples = random.sample(samples, num_shots)
        return samples
