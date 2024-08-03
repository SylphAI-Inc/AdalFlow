import csv
import json
import sys
import os
from typing import Literal
import subprocess
from lightrag.utils.data import Dataset
from dataclasses import dataclass, field
from lightrag.core.base_data_class import DataClass


@dataclass
class ObjectCountData(DataClass):
    x: str = field(metadata={"desc": "The question to be answered"})
    y: str = field(metadata={"desc": "The answer to the question"})


# TODO: here users clean adalflow created files
class BigBenchHard(Dataset):
    def __init__(
        self,
        task_name: Literal["BBH_object_counting"] = "BBH_object_counting",
        root: str = None,
        split: str = "train",
        *args,
        **kwargs,
    ):
        if root is None:
            # Set a default root directory based on the OS
            if sys.platform == "win32":
                root = os.path.join(os.getenv("APPDATA"), "adalflow")
            else:
                root = os.path.join(os.path.expanduser("~"), ".adalflow")
        self.root = root
        self.split = split
        self.task_name = "_".join(task_name.split("_")[1:])
        self._check_or_download_dataset()
        assert split in ["train", "val", "test"]
        # create the root directory if it does not exist
        os.makedirs(os.path.join(self.root, self.task_name), exist_ok=True)
        data_path = os.path.join(self.root, self.task_name, f"{split}.csv")
        self.data = []
        with open(data_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                print(row)
                self.data.append(
                    ObjectCountData(x=row["x"], y=row["y"])
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
        train_examples = [{"x": ex["input"], "y": ex["target"]} for ex in examples[:50]]
        val_examples = [
            {"x": ex["input"], "y": ex["target"]} for ex in examples[50:150]
        ]
        test_examples = [{"x": ex["input"], "y": ex["target"]} for ex in examples[150:]]

        for split, examples in zip(
            ["train", "val", "test"], [train_examples, val_examples, test_examples]
        ):
            with open(
                os.path.join(self.root, self.task_name, f"{split}.csv"), "w", newline=""
            ) as csvfile:
                fieldnames = ["x", "y"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(examples)

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
