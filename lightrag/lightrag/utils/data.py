from typing import Union, Tuple
import numpy as np
from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class DataLoader:
    __doc__ = r"""A simplified version of PyTorch DataLoader.

    The biggest difference is not to handle tensors, but to handle any type of data."""

    def __init__(self, dataset, batch_size: int = 4, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        self.current_index = 0

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_index = 0
        return self

    def __len__(self):
        return len(self.dataset) // self.batch_size + 1

    def __next__(self) -> Union[np.ndarray, Tuple]:
        if self.current_index >= len(self.dataset):
            raise StopIteration

        batch_indices = self.indices[
            self.current_index : self.current_index + self.batch_size
        ]
        batch_data = [self.dataset[int(i)] for i in batch_indices]

        if isinstance(batch_data[0], tuple):
            batch_data = tuple(zip(*batch_data))
        else:
            batch_data = np.array(batch_data)

        self.current_index += self.batch_size

        return batch_data


# concat two batches
def cat(batch1, batch2) -> Union[np.ndarray, Tuple]:
    if not batch1 or not batch2:
        return batch1 or batch2
    if isinstance(batch1, tuple):  # return tuple
        return tuple([np.concatenate([b1, b2]) for b1, b2 in zip(batch1, batch2)])
    else:
        return np.concatenate([batch1, batch2])
