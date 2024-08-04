import csv
import json
import os
import uuid
from typing import Literal
import subprocess
from lightrag.utils.data import Dataset
from dataclasses import dataclass, field
from lightrag.core.base_data_class import DataClass

from lightrag.utils.global_config import get_adalflow_default_root_path
from lightrag.utils.file_io import save_csv


@dataclass
class ObjectCountData(DataClass):
    id: str = field(
        metadata={"desc": "The unique identifier of the example"},
        default=str(uuid.uuid4()),
    )
    x: str = field(metadata={"desc": "The question to be answered"}, default=None)
    y: str = field(metadata={"desc": "The answer to the question"}, default=None)


# TODO: here users clean adalflow created files
class BigBenchHard(Dataset):
    def __init__(
        self,
        task_name: Literal["BBH_object_counting"] = "BBH_object_counting",
        root: str = None,
        split: Literal["train", "val", "test"] = "train",
        *args,
        **kwargs,
    ):

        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of 'train', 'val', 'test'")

        if root is None:
            root = get_adalflow_default_root_path()
            print(f"Saving dataset to {root}")
        self.root = root
        self.split = split
        self.task_name = "_".join(task_name.split("_")[1:])
        self._check_or_download_dataset()
        os.makedirs(os.path.join(self.root, self.task_name), exist_ok=True)
        data_path = os.path.join(self.root, self.task_name, f"{split}.csv")
        self.data = []
        with open(data_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.data.append(
                    ObjectCountData(x=row["x"], y=row["y"], id=row["id"])
                )  # dont use a tuple, use a dict {"x": ..., "y": ...}
        self._task_description = "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."

    def _check_or_download_dataset(self):
        data_path = os.path.join(self.root, self.task_name, f"{self.split}.csv")
        if os.path.exists(data_path):
            return

        os.makedirs(os.path.join(self.root, self.task_name), exist_ok=True)
        subprocess.call(
            [
                "wget",
                f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{self.task_name}.json",
                "-O",
                os.path.join(self.root, f"{self.task_name}.json"),
            ]
        )

        with open(os.path.join(self.root, f"{self.task_name}.json")) as json_file:
            data = json.load(json_file)

        examples = data["examples"]
        train_examples = [
            {"x": ex["input"], "y": ex["target"], "id": str(uuid.uuid4())}
            for ex in examples[:50]
        ]
        val_examples = [
            {"x": ex["input"], "y": ex["target"], "id": str(uuid.uuid4())}
            for ex in examples[50:150]
        ]
        test_examples = [
            {"x": ex["input"], "y": ex["target"], "id": str(uuid.uuid4())}
            for ex in examples[150:]
        ]

        for split, examples in zip(
            ["train", "val", "test"], [train_examples, val_examples, test_examples]
        ):
            target_path = os.path.join(self.root, self.task_name, f"{split}.csv")
            save_csv(examples, f=target_path, fieldnames=["x", "y", "id"])

    def __getitem__(self, index) -> ObjectCountData:
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_default_task_instruction(self):
        return self._task_description


if __name__ == "__main__":
    dataset = BigBenchHard(
        "BBH_object_counting", split="train", root="BBH_object_counting"
    )
    print(dataset[0])
    print(len(dataset))
    print(dataset.get_default_task_instruction())
