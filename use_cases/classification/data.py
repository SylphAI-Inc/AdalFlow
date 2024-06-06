# https://huggingface.co/datasets/trec
# labels: https://huggingface.co/datasets/trec/blob/main/trec.py
from typing import Sequence, Dict
import re
import os

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch

from datasets import load_dataset, DatasetDict, load_from_disk
from datasets import Dataset as HFDataset


from lightrag.core.prompt_builder import Prompt
from lightrag.core.component import Component
from lightrag.optim.sampler import Sample, ClassSampler
from lightrag.utils import save, load

from .utils import get_script_dir


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
EXAMPLES_STR = r"""Question: {{input}}
{%if thought%}
thought: {{thought}} 
{%endif%}
class_name: {{output}} 
{%if description%}({{description}}){%endif%}
class_index: {{label}}
--------
"""


class SamplesToStr(Component):
    # TODO: make the samples to only input and output data
    def __init__(self):
        super().__init__()
        self.template = Prompt(template=EXAMPLES_STR)

    def call_one(self, sample: Sample) -> str:
        data = sample.data
        assert "text" in data, "The data must have a 'text' field"
        assert "coarse_label" in data, "The data must have a 'coarse_label' field"
        example_str = self.template(
            input=data["text"],
            label=data["coarse_label"],
            output=_COARSE_LABELS_DESC[data["coarse_label"]],
            thought=data.get("thought", None),
        )
        return example_str

    def call(self, samples: Sequence[Sample]) -> str:
        return "\n".join([self.call_one(sample) for sample in samples])


class TrecDataset(Dataset):
    r"""
    Juse one example for customizing the dataset. Not used in this use case.
    """

    def __init__(self, dataset: DatasetDict, split: str):
        """
        Args:
            dataset: The dataset to use.
            split: The split to use.
        """
        self.dataset = dataset[split]
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        r"""
        Return the trainable states used as the input to the model [task pipeline]
        """
        # Retrieve the data at the specified index
        data = self.dataset[idx]
        return data


def extract_class_label(text: str) -> int:
    re_pattern = r"\d+"
    if isinstance(text, str):
        label_match = re.findall(re_pattern, text)
        if label_match:
            label = int(label_match[0])
        else:
            label = -1
        return label
    else:
        return text


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


# make sure to run this only once to prepare a small set of train, eval, and test datasets and keep it fixed during different experiments.
def prepare_datasets(path: str = None):
    path = os.path.join(get_script_dir(), "data") or path
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

    # test_sampler = ClassSampler(
    #     org_test_dataset, num_classes=6, get_data_key_fun=lambda x: x["coarse_label"]
    # )
    # test_dataset_split = [sample.data for sample in test_sampler(test_size)]

    print(
        f"train example: {train_dataset_split[0]}, type: {type(train_dataset_split[0])}"
    )

    # save the datasets in the data folder
    train_dataset_split.save_to_disk(
        f"{path}/train"
    )  # TODO: update the dataset info to the new dataset

    # use json to save for better readability
    # along with pickle for easy loading
    save(
        eval_dataset_split,
        f"{path}/eval",
    )
    save(
        test_dataset_split,
        f"{path}/test",
    )


def load_datasets(path: str = None):
    path = os.path.join(get_script_dir(), "data") or path
    train_dataset: HFDataset = load_from_disk(dataset_path=f"{path}/train")
    eval_dataset = load(f"{path}/eval")[1]
    test_dataset = load(f"{path}/test")[1]
    print(f"train: {len(train_dataset)}")
    print(f"eval: {len(eval_dataset)}")
    print(f"test: {len(test_dataset)}")

    return train_dataset, eval_dataset, test_dataset


if __name__ == "__main__":
    load_datasets()
