import csv
import json
import os
import uuid
from typing import Literal
import subprocess
from adalflow.utils.data import Dataset
from adalflow.datasets.types import Example

from adalflow.utils.global_config import get_adalflow_default_root_path
from adalflow.utils.file_io import save_csv


def prepare_dataset_path(root: str, task_name: str):
    if root is None:
        root = os.path.join(get_adalflow_default_root_path(), "cache_datasets")

    save_path = os.path.join(root, task_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path


# TODO: here users clean adalflow created files
class BigBenchHard(Dataset):
    __doc__ = """Big Bench Hard dataset for object counting task.

    Data will be saved to ~/.adalflow/cache_datasets/BBH_object_counting/{split}.csv
    if root is not specified.

    Size for each split:
    - train: 50 examples
    - val: 50 examples
    - test: 100 examples
    """

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

        # if root is None:
        #     root = get_adalflow_default_root_path()
        #     print(f"Saving dataset to {root}")
        self.root = root
        self.split = split

        self.task_name = "_".join(task_name.split("_")[1:])
        data_path = prepare_dataset_path(self.root, self.task_name)
        self._check_or_download_dataset(data_path, split)

        self.data = []
        split_csv_path = os.path.join(data_path, f"{split}.csv")
        with open(split_csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.data.append(
                    Example(question=row["x"], answer=row["y"], id=row["id"])
                )  # dont use a tuple, use a dict {"x": ..., "y": ...}
        self._task_description = "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."

    def _check_or_download_dataset(self, data_path: str = None, split: str = "train"):

        if data_path is None:
            raise ValueError("data_path must be specified")
        json_path = os.path.join(data_path, f"{self.task_name}.json")
        split_csv_path = os.path.join(data_path, f"{split}.csv")
        if os.path.exists(split_csv_path):
            return

        print(f"Downloading dataset to {json_path}")

        subprocess.call(
            [
                "wget",
                f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{self.task_name}.json",
                "-O",
                json_path,
            ]
        )

        with open(json_path) as json_file:
            data = json.load(json_file)

        examples = data["examples"]

        train_examples = [
            {"x": ex["input"], "y": ex["target"], "id": str(uuid.uuid4())}
            for ex in examples[:50]
        ]
        val_examples = [
            {"x": ex["input"], "y": ex["target"], "id": str(uuid.uuid4())}
            for ex in examples[50:100]
        ]
        test_examples = [
            {"x": ex["input"], "y": ex["target"], "id": str(uuid.uuid4())}
            for ex in examples[100:200]
        ]

        for split, examples in zip(
            ["train", "val", "test"], [train_examples, val_examples, test_examples]
        ):
            target_path = os.path.join(data_path, f"{split}.csv")
            print(f"Saving {split} split to {target_path}")
            save_csv(examples, f=target_path, fieldnames=["x", "y", "id"])

    def __getitem__(self, index) -> Example:
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_default_task_instruction(self):
        return self._task_description


if __name__ == "__main__":
    from adalflow.datasets.big_bench_hard import BigBenchHard

    dataset = BigBenchHard("BBH_object_counting", split="train")
    print(dataset[0])
    print(len(dataset))
    print(dataset.get_default_task_instruction())
