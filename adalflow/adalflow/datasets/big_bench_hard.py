import csv
import json
import os
import uuid
from typing import Literal
import urllib.request
import urllib.error
from adalflow.utils.data import Dataset
from adalflow.datasets.types import Example

from adalflow.utils.file_io import save_csv
from adalflow.datasets.utils import prepare_dataset_path


# TODO: here users clean adalflow created files
class BigBenchHard(Dataset):
    __doc__ = """Big Bench Hard dataset for object counting task.

    You can find the task name from the following link:
    https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh


    Data will be saved to ~/.adalflow/cache_datasets/BBH_object_counting/{split}.csv
    if root is not specified.

    Size for each split:
    - train: 50 examples
    - val: 100 examples
    - test: 100 examples

    Args:
        task_name (str): The name of the task. "{task_name}" is the task name in the dataset.
        root (str, optional): Root directory of the dataset to save the data. Defaults to ~/.adalflow/cache_datasets/task_name.
        split (str, optional): The dataset split, supports ``"train"`` (default), ``"val"`` and ``"test"``.
    """

    def __init__(
        self,
        task_name: Literal["object_counting"] = "object_counting",
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

        self.task_name = task_name
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

    def _check_or_download_dataset(self, data_path: str = None, split: str = "train"):

        if data_path is None:
            raise ValueError("data_path must be specified")
        json_path = os.path.join(data_path, f"{self.task_name}.json")
        split_csv_path = os.path.join(data_path, f"{split}.csv")
        if os.path.exists(split_csv_path):
            return

        print(f"Downloading dataset to {json_path}")
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(json_path), exist_ok=True)

            # Use urllib.request instead of wget for cross-platform compatibility
            url = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{self.task_name}.json"
            urllib.request.urlretrieve(url, json_path)

            # Check if the file is non-empty
            if not os.path.exists(json_path) or os.path.getsize(json_path) == 0:
                raise ValueError(
                    f"Downloaded file is empty. Please check the task name '{self.task_name}' or network issues."
                )

        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise ValueError(
                    f"Task name '{self.task_name}' not found (HTTP 404).\n"
                    "Please verify the task name (the JSON file name) by checking the following link:\n"
                    "https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh"
                ) from e
            else:
                raise ValueError(
                    f"Failed to download dataset for task '{self.task_name}' (HTTP {e.code}).\n"
                    "Please check your internet connection or try again later."
                ) from e
        except urllib.error.URLError as e:
            raise ValueError(
                f"Network error while downloading dataset for task '{self.task_name}'.\n"
                "Please check your internet connection and try again."
            ) from e
        except Exception as e:
            raise ValueError(
                f"Unexpected error while downloading dataset for task '{self.task_name}': {str(e)}\n"
                "Please verify the task name by checking the following link:\n"
                "https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh"
            ) from e

        with open(json_path) as json_file:
            data = json.load(json_file)

        examples = data["examples"]

        # NOTE: better to shuffle the examples before splitting.
        # We do this splitting in order to be consistent with text-grad paper.

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
        # ensure the

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

    @staticmethod
    def get_default_task_instruction():
        _task_description = "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."
        return _task_description


if __name__ == "__main__":
    from adalflow.datasets.big_bench_hard import BigBenchHard

    dataset = BigBenchHard(task_name="object_counting", split="test")
    print(dataset[0:10])
    print(len(dataset))
    print(dataset.get_default_task_instruction())
