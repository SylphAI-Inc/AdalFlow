import random
import os
from typing import Literal
import tqdm

from adalflow.utils.lazy_import import safe_import, OptionalPackages


from adalflow.utils.data import Dataset
from adalflow.utils.file_io import save_json, load_json
from adalflow.datasets.utils import prepare_dataset_path
from adalflow.core.base_data_class import DataClass
from adalflow.datasets.types import GSM8KData
from adalflow.utils import printc


class GSM8K(Dataset):
    __doc__ = r""" Use huggingface datasets to load GSM8K dataset.

            official_train: 7473
            official_test: 1319

            Our train split: 3736/2
            Our val split: 3736/2
            Our test split: 1319

        You can use size to limit the number of examples to load.

        Example:

        .. code-block:: python

            dataset = GSM8K(split="train", size=10)

            print(f"example: {dataset[0]}")

        The output will be:

        .. code-block::

            GSM8KData(id='8fc791e6-ea1d-472c-a882-d00d0600d423',
            question="The result from the 40-item Statistics exam Marion and Ella took already came out.
            Ella got 4 incorrect answers while Marion got 6 more than half the score of Ella.
              What is Marion's score?",
              answer='24',
              gold_reasoning="Ella's score is 40 items - 4 items = <<40-4=36>>36 items.
              Half of Ella's score is 36 items / 2 = <<36/2=18>>18 items.
              So, Marion's score is 18 items + 6 items = <<18+6=24>>24 items.",
              reasoning=None)
        """

    def __init__(
        self,
        root: str = None,
        split: Literal["train", "val", "test"] = "train",
        size: int = None,
        **kwargs,
    ) -> None:

        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of 'train', 'val', 'test'")

        self.root = root
        self.task_name = "gsm8k"
        data_path = prepare_dataset_path(self.root, self.task_name)
        # download and save
        split_csv_path = os.path.join(data_path, f"{split}.json")
        print(f"split_csv_path: {split_csv_path}")
        self._check_or_download_dataset(split_csv_path, split)

        # load from csv
        self.data = []

        self.data = load_json(split_csv_path)
        if size is not None:
            self.data = self.data[:size]
        # convert to dataclass
        self.data = [GSM8KData.from_dict(d) for d in self.data]

    def _check_or_download_dataset(
        self,
        data_path: str = None,
        split: str = "train",
    ):
        r"""It will download data from huggingface datasets and split it and save it into three csv files.
        Args:
            data_path (str): The path to save the data. In particular with split name appended.
            split (str): The dataset split, supports ``"train"`` (default), ``"val"`` and ``"test"``. Decides which split to return.
            only_hard_examples (bool): If True, only hard examples will be downloaded.
            keep_details (str): If "all", all details will be kept. If "dev_titles", only dev titles will be kept.
        """

        if data_path is None:
            raise ValueError("data_path must be specified")

        if os.path.exists(data_path):
            return

        safe_import(
            OptionalPackages.DATASETS.value[0], OptionalPackages.DATASETS.value[1]
        )
        from datasets import load_dataset

        # use huggingface cache
        gsm8k_dataset = load_dataset("gsm8k", "main", cache_dir=self.root)

        hf_official_train = gsm8k_dataset["train"]
        hf_official_test = gsm8k_dataset["test"]

        official_train = []
        official_test = []

        for example in tqdm.tqdm(hf_official_train):
            question = example["question"]
            answer = example["answer"].strip().split()
            assert answer[-2] == "####"

            gold_reasoning = " ".join(answer[:-2])
            answer = str(int(answer[-1].replace(",", "")))
            official_train.append(
                dict(question=question, gold_reasoning=gold_reasoning, answer=answer)
            )

        for example in tqdm.tqdm(hf_official_test):
            question = example["question"]
            answer = example["answer"].strip().split()
            assert answer[-2] == "####"

            gold_reasoning = " ".join(answer[:-2])
            answer = str(int(answer[-1].replace(",", "")))
            official_test.append(
                dict(question=question, gold_reasoning=gold_reasoning, answer=answer)
            )

        rng = random.Random(0)
        rng.shuffle(official_train)  # 7473 train
        rng = random.Random(0)
        rng.shuffle(official_test)  # 1319 test

        printc(f"official_train: {len(official_train)}")
        printc(f"official_test: {len(official_test)}")
        train_set = official_train[: len(official_train) * 50 // 100]
        val_set = official_train[len(official_train) * 50 // 100 :]
        data_path_dir = os.path.dirname(data_path)
        for split, examples in zip(
            ["train", "val", "test"],
            [train_set, val_set, official_test],
        ):
            target_path = os.path.join(data_path_dir, f"{split}.json")
            save_json(examples, f=target_path)

        if split == "train":
            return train_set
        elif split == "val":
            return val_set
        else:
            return official_test

    def __getitem__(self, index) -> DataClass:
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = GSM8K(split="train", size=10)

    print(f"len: {len(dataset)}")
    print(f"dataset[0]: {dataset[0]}")
