import random
import os
import csv
from typing import Literal

from adalflow.utils.lazy_import import safe_import, OptionalPackages


from adalflow.utils.data import Dataset
from adalflow.utils.file_io import save_csv
from adalflow.datasets.utils import prepare_dataset_path
from adalflow.core.base_data_class import DataClass
from adalflow.datasets.types import HotPotQAData


class HotPotQA(Dataset):
    def __init__(
        self,
        only_hard_examples=True,
        root: str = None,
        split: Literal["train", "val", "test"] = "train",
        keep_details: Literal["all", "dev_titles", "none"] = "dev_titles",
        size: int = None,
        **kwargs,
    ) -> None:
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of 'train', 'val', 'test'")

        if keep_details not in ["all", "dev_titles", "none"]:
            raise ValueError("Keep details must be one of 'all', 'dev_titles', 'none'")

        # if root is None:
        #     root = get_adalflow_default_root_path()
        #     print(f"Saving dataset to {root}")
        self.root = root
        self.task_name = f"hotpot_qa_{keep_details}"
        data_path = prepare_dataset_path(self.root, self.task_name)
        # download and save
        split_csv_path = os.path.join(data_path, f"{split}.csv")
        self._check_or_download_dataset(
            split_csv_path, split, only_hard_examples, keep_details
        )

        # load from csv
        self.data = []
        # created_data_class = DynamicDataClassFactory.from_dict(
        #  "HotPotQAData", {"id": "str", "question": "str", "answer": "str"}

        with open(split_csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                if size is not None and i >= size:
                    break
                self.data.append(HotPotQAData.from_dict(row))

    def _check_or_download_dataset(
        self,
        data_path: str = None,
        split: str = "train",
        only_hard_examples=True,
        keep_details="dev_titles",
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

        assert only_hard_examples, (
            "Care must be taken when adding support for easy examples."
            "Dev must be all hard to match official dev, but training can be flexible."
        )

        hf_official_train = load_dataset(
            "hotpot_qa", "fullwiki", split="train", trust_remote_code=True
        )
        hf_official_dev = load_dataset(
            "hotpot_qa", "fullwiki", split="validation", trust_remote_code=True
        )
        keys = ["question", "answer"]
        if keep_details == "all":
            keys = [
                "id",
                "question",
                "answer",
                "type",
                "supporting_facts",
                "context",
            ]
        elif keep_details == "dev_titles":
            keys = ["id", "question", "answer", "supporting_facts"]

        official_train = []
        for raw_example in hf_official_train:
            if raw_example["level"] == "hard":
                example = {k: raw_example[k] for k in keys}

                if "supporting_facts" in example:
                    example["gold_titles"] = set(example["supporting_facts"]["title"])
                    del example["supporting_facts"]

                official_train.append(example)

        rng = random.Random(0)
        rng.shuffle(official_train)

        sampled_trainset = official_train[: len(official_train) * 75 // 100]

        sampled_valset = official_train[
            len(official_train) * 75 // 100 :
        ]  # this is not the official dev set

        # for example in self._train:
        #     if keep_details == "dev_titles":
        #         del example["gold_titles"]

        test = []
        for raw_example in hf_official_dev:
            assert raw_example["level"] == "hard"
            example = {
                k: raw_example[k]
                for k in ["id", "question", "answer", "type", "supporting_facts"]
            }
            if "supporting_facts" in example:
                example["gold_titles"] = set(example["supporting_facts"]["title"])
                del example["supporting_facts"]
            test.append(example)

        keys = ["id", "question", "answer", "gold_titles"]
        # save to csv
        for split, examples in zip(
            ["train", "val", "test"],
            [sampled_trainset, sampled_valset, test],
        ):
            # target_path = prepare_dataset_path(self.root, task_name, split)
            save_csv(examples, f=data_path, fieldnames=keys)

        if split == "train":
            return sampled_trainset
        elif split == "val":
            return sampled_valset
        else:
            return test

    def __getitem__(self, index) -> DataClass:
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = HotPotQA(root="BBH_object_counting", split="train", size=20)
    print(dataset[0], type(dataset[0]))
    print(len(dataset))
    valdataset = HotPotQA(root="BBH_object_counting", split="val", size=50)
    print(len(valdataset))
    testdataset = HotPotQA(root="BBH_object_counting", split="test", size=50)
    print(len(testdataset))
    print(f"valdataset[0]: {valdataset[0]}")
    print(f"testdataset[0]: {testdataset[0]}")
