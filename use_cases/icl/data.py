# https://huggingface.co/datasets/trec
# labels: https://huggingface.co/datasets/trec/blob/main/trec.py

from datasets import load_dataset, DatasetDict

_COARSE_LABELS = [
    "ABBR:Abbreviation",
    "ENTY:Entity",
    "DESC:Description and abstract concept",
    "HUM:Human being",
    "LOC:Location",
    "NUM:Numeric value",
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

import torch
from typing import Dict, Sequence


class TrecDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: DatasetDict, split: str):
        """
        Args:
            dataset: The dataset to use.
            split: The split to use.
        """
        self.dataset = dataset
        self.split = split

    def __len__(self):
        return len(self.dataset[self.split])

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item["text"]
        label = item["label-coarse"]
        inputs = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs["label"] = torch.tensor(_COARSE_LABELS.index(label))
        return inputs
