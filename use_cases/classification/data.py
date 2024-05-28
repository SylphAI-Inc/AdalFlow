# https://huggingface.co/datasets/trec
# labels: https://huggingface.co/datasets/trec/blob/main/trec.py

from datasets import load_dataset, DatasetDict
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Sequence

from core.prompt_builder import Prompt
from core.component import Component
from typing import Any

from use_cases.classification.prompt import EXAMPLES_STR
from optim.sampler import Sample

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

_FINE_LABELS = [
    "ABBR:abb",
    "ABBR:exp",
    "ENTY:animal",
    "ENTY:body",
    "ENTY:color",
    "ENTY:cremat",
    "ENTY:currency",
    "ENTY:dismed",
    "ENTY:event",
    "ENTY:food",
    "ENTY:instru",
    "ENTY:lang",
    "ENTY:letter",
    "ENTY:other",
    "ENTY:plant",
    "ENTY:product",
    "ENTY:religion",
    "ENTY:sport",
    "ENTY:substance",
    "ENTY:symbol",
    "ENTY:techmeth",
    "ENTY:termeq",
    "ENTY:veh",
    "ENTY:word",
    "DESC:def",
    "DESC:desc",
    "DESC:manner",
    "DESC:reason",
    "HUM:gr",
    "HUM:ind",
    "HUM:title",
    "HUM:desc",
    "LOC:city",
    "LOC:country",
    "LOC:mount",
    "LOC:other",
    "LOC:state",
    "NUM:code",
    "NUM:count",
    "NUM:date",
    "NUM:dist",
    "NUM:money",
    "NUM:ord",
    "NUM:other",
    "NUM:period",
    "NUM:perc",
    "NUM:speed",
    "NUM:temp",
    "NUM:volsize",
    "NUM:weight",
]


dataset = load_dataset("trec")
print(dataset)

print(f"Train example: {dataset['train'][0]}")
print(f"Test example: {dataset['test'][0]}")


class SamplesToStr(Component):
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
            # description=_COARSE_LABELS_DESC[int(data["coarse_label"])],
        )
        # example_str = "*" * len(example_str)
        return example_str

    def call(self, samples: Sequence[Sample]) -> str:
        return "\n".join([self.call_one(sample) for sample in samples])


# class ToSampleStr(Component):
#     def __init__(self):
#         super().__init__()
#         self.template = Prompt(template=EXAMPLES_STR)

#     def call(self, sample: Sample) -> str:
#         data = sample.data
#         assert "text" in data, "The data must have a 'text' field"
#         assert "coarse_label" in data, "The data must have a 'coarse_label' field"
#         example_str = self.template(
#             input=data["text"],
#             label=data["coarse_label"],
#             output=_COARSE_LABELS_DESC[data["coarse_label"]],
#             # description=_COARSE_LABELS_DESC[int(data["coarse_label"])],
#         )
#         # example_str = "*" * len(example_str)
#         return example_str


class TrecDataset(Dataset):
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


# dataset = CIFAR100(transform=None, download=True)
# data_loader = DataLoader(dataset["train"], batch_size=2, shuffle=True)
# for batch in data_loader:
#     print(batch)
#     print(batch["text"], batch["coarse_label"], batch["fine_label"])
#     break
import re


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
