import os
import csv
from typing import Literal, Dict

from adalflow.utils.lazy_import import safe_import, OptionalPackages

safe_import(OptionalPackages.TORCH.value[0], OptionalPackages.TORCH.value[1])
safe_import(OptionalPackages.DATASETS.value[0], OptionalPackages.DATASETS.value[1])

import torch
from torch.utils.data import WeightedRandomSampler
from datasets import Dataset as HFDataset

from adalflow.utils.data import Dataset
from adalflow.utils.global_config import get_adalflow_default_root_path
from adalflow.utils.file_io import save_csv
from adalflow.datasets.big_bench_hard import prepare_dataset_path
from adalflow.datasets.types import TrecData


def calculate_class_weights(labels: torch.Tensor) -> torch.Tensor:
    # Count frequencies of each class
    class_counts = torch.bincount(labels)
    # Calculate weight for each class (inverse frequency)
    class_weights = 1.0 / class_counts.float()
    # Assign weight to each sample
    sample_weights = class_weights[labels]
    return sample_weights


def sample_subset_dataset(
    dataset: HFDataset, num_samples: int, sample_weights
) -> HFDataset:
    # Create a WeightedRandomSampler to get 400 samples
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=num_samples, replacement=False
    )

    # Extract indices from the sampler
    indices = list(iter(sampler))
    # Create a subset of the dataset
    subset_dataset = dataset.select(indices)
    return subset_dataset


def prepare_datasets():
    from datasets import load_dataset
    from datasets import Dataset as HFDataset
    from adalflow.optim.sampler import ClassSampler

    dataset = load_dataset("trec")
    print(f"train: {len(dataset['train'])}, test: {len(dataset['test'])}")  # 5452, 500
    print(f"train example: {dataset['train'][0]}")

    num_classes = 6

    # (1) create eval dataset from the first 1/3 of the train datset, 6 samples per class
    org_train_dataset = dataset["train"].shuffle(seed=42)
    train_size = num_classes * 100
    len_train_dataset = len(org_train_dataset)

    org_test_dataset = dataset["test"]
    eval_size = 6 * num_classes

    class_sampler = ClassSampler(
        org_train_dataset.select(
            range(0, len_train_dataset // 3)
        ),  # created huggingface dataset type
        num_classes=num_classes,
        get_data_key_fun=lambda x: x["coarse_label"],
    )

    eval_dataset_split = [sample.data for sample in class_sampler(eval_size)]
    # convert this back to huggingface dataset
    eval_dataset_split = HFDataset.from_list(eval_dataset_split)

    # (2) create train dataset from the last 2/3 of the train dataset, 100 samples per class
    train_dataset_split = org_train_dataset.select(
        range(len_train_dataset // 3, len_train_dataset)
    )  # {4: 413, 5: 449, 1: 630, 2: 560, 3: 630, 0: 44}
    labels = torch.tensor(train_dataset_split["coarse_label"])
    class_weights = calculate_class_weights(labels)
    print(f"class_weights: {class_weights}")

    train_dataset_split = sample_subset_dataset(
        train_dataset_split, train_size, class_weights
    )
    print(f"train example: {train_dataset_split[0]}")
    print(f"train: {len(train_dataset_split)}, eval: {len(eval_dataset_split)}")

    # get the count for each class
    count_by_class: Dict[str, int] = {}
    for sample in train_dataset_split:
        label = sample["coarse_label"]
        count_by_class[label] = count_by_class.get(label, 0) + 1

    print(f"count_by_class: {count_by_class}")

    # create the test dataset from the test dataset
    # weights for the test dataset
    labels = torch.tensor(org_test_dataset["coarse_label"])
    class_weights = calculate_class_weights(labels)

    test_size = eval_size * 4
    # weighted sampling on the test dataset
    test_dataset_split = sample_subset_dataset(
        org_test_dataset, test_size, class_weights
    )

    print(
        f"train example: {train_dataset_split[0]}, type: {type(train_dataset_split[0])}"
    )
    return train_dataset_split, eval_dataset_split, test_dataset_split


_COARSE_LABELS = [
    "ABBR",
    "ENTY",
    "DESC",
    "HUM",
    "LOC",
    "NUM",
]

_COARSE_LABELS_DESC = [
    "Abbreviation",
    "Entity",
    "Description and abstract concept",
    "Human being",
    "Location",
    "Numeric value",
]


class TrecDataset(Dataset):
    __doc__ = r"""Trec dataset for question classification.


    Here we only load a small subset of the dataset for training and evaluation.

    In default: train: 600, 100 per class, val: 36, test: 144
    All class-balanced.

    Reference:
    - https://huggingface.co/datasets/trec
    labels: https://huggingface.co/datasets/trec/blob/main/trec.py"""

    def __init__(
        self, root: str = None, split: Literal["train", "test"] = "train"
    ) -> None:
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of 'train', 'val', 'test'")
        if root is None:
            root = get_adalflow_default_root_path()
            print(f"Saving dataset to {root}")
        self.root = root
        self.task_name = "trec_classification"
        data_path = prepare_dataset_path(self.root, self.task_name, split)
        # download and save
        self._check_or_download_dataset(data_path, split)
        # load from csv
        self.data = []
        with open(data_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.data.append(
                    TrecData(
                        id=row["id"],
                        question=row["text"],
                        class_index=int(row["coarse_label"]),
                        class_name=_COARSE_LABELS_DESC[int(row["coarse_label"])],
                    )
                )

    def _check_or_download_dataset(self, data_path: str = None, split: str = "train"):
        import uuid

        if os.path.exists(data_path):
            return
        # prepare all the data
        train_dataset, val_dataset, test_dataset = prepare_datasets()
        # save to csv
        keys = ["id", "text", "coarse_label"]
        for split, examples in zip(
            ["train", "val", "test"],
            [train_dataset, val_dataset, test_dataset],
        ):
            # add ids to the examples
            new_examples = []
            for i, example in enumerate(examples):
                example["id"] = str(uuid.uuid4())
                new_examples.append(example)

            target_path = prepare_dataset_path(self.root, self.task_name, split)
            save_csv(new_examples, f=target_path, fieldnames=keys)

        # Return the dataset to data
        if split == "train":
            return train_dataset
        elif split == "val":
            return val_dataset
        else:
            return test_dataset

    def __getitem__(self, index) -> TrecData:
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = TrecDataset(split="val")
    print(f"train: {len(dataset)}, example: {dataset[0]}")
