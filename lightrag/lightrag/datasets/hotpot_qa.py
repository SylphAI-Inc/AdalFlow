import random
from typing import Literal

from lightrag.utils.lazy_import import safe_import, OptionalPackages


datasets = safe_import(
    OptionalPackages.DATASETS.value[0], OptionalPackages.DATASETS.value[1]
)
from datasets import load_dataset

from lightrag.utils.data import Dataset
from lightrag.utils.global_config import get_adalflow_default_root_path


class HotPotQA(Dataset):
    def __init__(
        self,
        only_hard_examples=True,
        root: str = None,
        split: Literal["train", "val", "test"] = "train",
        keep_details: Literal["all", "dev_titles", "none"] = "dev_titles",
        **kwargs,
    ) -> None:
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of 'train', 'val', 'test'")

        if keep_details not in ["all", "dev_titles", "none"]:
            raise ValueError("Keep details must be one of 'all', 'dev_titles', 'none'")

        if root is None:
            root = get_adalflow_default_root_path()
            print(f"Saving dataset to {root}")
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
            keys = ["question", "answer", "supporting_facts"]

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

        if split == "train":
            return sampled_trainset
        elif split == "val":
            return sampled_valset
        else:
            return test

        # save to csv
        # use a data structure to load it
