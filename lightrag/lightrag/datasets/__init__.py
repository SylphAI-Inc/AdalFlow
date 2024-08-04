import random

from datasets import load_dataset

from dspy.datasets.dataset import Dataset


class HotPotQA(Dataset):
    def __init__(
        self,
        *args,
        only_hard_examples=True,
        keep_details="dev_titles",
        unofficial_dev=True,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
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

        official_train = []
        for raw_example in hf_official_train:
            if raw_example["level"] == "hard":
                if keep_details is True:
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
                else:
                    keys = ["question", "answer"]

                example = {k: raw_example[k] for k in keys}

                if "supporting_facts" in example:
                    example["gold_titles"] = set(example["supporting_facts"]["title"])
                    del example["supporting_facts"]

                official_train.append(example)

        rng = random.Random(0)
        rng.shuffle(official_train)

        self._train = official_train[: len(official_train) * 75 // 100]

        if unofficial_dev:
            self._dev = official_train[len(official_train) * 75 // 100 :]
        else:
            self._dev = None

        for example in self._train:
            if keep_details == "dev_titles":
                del example["gold_titles"]

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

        self._test = test
