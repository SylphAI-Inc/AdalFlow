"""Default Dataset, DataLoader similar to `utils.data` in PyTorch.

You can also use those provided by PyTorch or huggingface/datasets."""

from typing import Union, Tuple, List, Sequence, TypeVar, Generic
import numpy as np
import random

T_co = TypeVar("T_co", covariant=True)


# TODO: consider directly use torch.utils.data in the future
class Dataset(Generic[T_co]):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`. Subclasses could also
    optionally implement :meth:`__getitems__`, for speedup batched samples
    loading. This method accepts list of indices of samples of batch and returns
    list of samples.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs an index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class Subset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: List[int]) -> List[T_co]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            return [self.dataset[self.indices[idx]] for idx in indices]

    def __len__(self):
        return len(self.indices)


class DataLoader:
    __doc__ = r"""A simplified version of PyTorch DataLoader.

    The biggest difference is not to handle tensors, but to handle any type of data."""

    def __init__(self, dataset, batch_size: int = 4, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = np.arange(len(dataset))
        # if self.shuffle:
        #     np.random.shuffle(self.indices)
        self.current_index = 0
        self.max_steps = self.__len__()
        self.step_index = 0

    def set_max_steps(self, max_steps: int):
        self.max_steps = max_steps

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_index = 0
        return self

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __next__(self) -> Union[np.ndarray, Tuple]:
        # if self.current_index >= len(self.dataset):
        #     raise StopIteration

        if self.current_index >= len(self.dataset):
            if self.shuffle:
                np.random.shuffle(self.indices)  # Reshuffle for the new epoch
            self.current_index = 0
            if self.step_index < self.max_steps:
                pass
            else:
                raise StopIteration
            # raise StopIteration

        batch_indices = self.indices[
            self.current_index : self.current_index + self.batch_size
        ]
        batch_data = [self.dataset[int(i)] for i in batch_indices]

        if isinstance(batch_data[0], tuple):
            batch_data = tuple(zip(*batch_data))
        else:
            batch_data = np.array(batch_data)

        self.current_index += self.batch_size
        self.step_index += 1

        return batch_data


def subset_dataset(dataset, num_samples: int):
    r"""This function will be useful for testing and debugging purposes."""
    num_samples = min(num_samples, len(dataset))
    random_subset_indices = random.sample(range(len(dataset)), num_samples)
    return Subset(dataset, random_subset_indices)
