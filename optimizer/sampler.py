import random
from typing import List, Sequence, Optional, Callable, Any, Tuple
import math

from core.component import Component


class ClassSampler(Component):
    def __init__(
        self,
        dataset,
        num_classes: int,
        get_data_key_fun: Callable,
    ):
        super().__init__()
        self.dataset = dataset
        self.num_classes = num_classes
        if not get_data_key_fun:
            raise ValueError("get_data_key_fun must be provided")
        self.get_key = get_data_key_fun
        self.class_indices: List[List] = [[] for _ in range(num_classes)]
        for i, data in enumerate(dataset):
            self.class_indices[self.get_key(data)].append(i)

    def _sample_one_class(self, num_samples: int, class_index: int) -> List[Any]:
        r"""
        Sample num_samples from the class with class_index"""
        incides = random.sample(self.class_indices[class_index], num_samples)
        samples = [self.dataset[i] for i in incides]
        return samples

    def random_replace(self, num_shots: int, samples: List[Any]):
        r"""
        Randomly select num_shots from the samples and replace it with another sample has the same class index
        """
        assert (
            len(samples) >= num_shots
        ), "num_shots is larger than the number of samples"
        # select num_shots in self.current to replace
        indices = random.sample(range(len(samples)), num_shots)
        for i in indices:
            class_index = self.get_key(samples[i])
            samples[i] = self._sample_one_class(1, class_index)[0]
        return samples

    def sample_one_class_with_indice(
        self, num_samples: int, class_index: int
    ) -> List[Any]:
        r"""
        Sample num_samples from the class with class_index"""
        incides = random.sample(self.class_indices[class_index], num_samples)
        samples = [(i, self.dataset[i]) for i in incides]
        return samples

    def call(self, shots: int) -> Sequence[str]:
        samples = []
        samples_per_class = math.ceil(shots / self.num_classes)
        for class_index in range(self.num_classes):
            samples.extend(self._sample_one_class(samples_per_class, class_index))
        if len(samples) > shots:
            # randomly sample from the class balance the number of samples
            samples = random.sample(samples, shots)
        return samples


class RandomSampler(Component):
    def __init__(self, dataset, num_shots: Optional[int] = None):
        super().__init__()
        self.dataset = dataset
        self.num_shots = num_shots

    def __call__(self, num_shots: Optional[int] = None) -> Sequence[str]:
        if num_shots is None:
            num_shots = self.num_shots
        if num_shots is None:
            raise ValueError("num_shots is not set")
        indices = random.sample(range(len(self.dataset)), num_shots)
        return [self.dataset[i] for i in indices]
