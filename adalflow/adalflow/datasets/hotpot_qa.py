import random
import os
from typing import Literal, List

from adalflow.utils.lazy_import import safe_import, OptionalPackages


from adalflow.utils.data import Dataset
from adalflow.utils.file_io import save_csv, save_json, load_json
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
        r"""
        official_train: 15661
        sampled_trainset: 11745
        sampled_valset: 3916
        test: 7405

        All answers are a phrase in the supporting context where we can choose supporting facts from the context.

        You can specify the size of the dataset to load by setting the size parameter.
        """
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
        split_csv_path = os.path.join(data_path, f"{split}.json")
        print(f"split_csv_path: {split_csv_path}")
        self._check_or_download_dataset(
            split_csv_path, split, only_hard_examples, keep_details
        )

        # load from csv
        self.data = []
        # created_data_class = DynamicDataClassFactory.from_dict(
        #  "HotPotQAData", {"id": "str", "question": "str", "answer": "str"}

        # with open(split_csv_path, newline="") as csvfile:
        #     reader = csv.DictReader(csvfile)
        #     for i, row in enumerate(reader):
        #         if size is not None and i >= size:
        #             break
        #         self.data.append(HotPotQAData.from_dict(row))

        self.data = load_json(split_csv_path)
        if size is not None:
            # use random seed to make sure the same data is loaded
            # random.Random(0).shuffle(self.data)
            self.data = self.data[:size]
        # convert to dataclass
        self.data = [HotPotQAData.from_dict(d) for d in self.data]

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

        # hf_official_train = load_dataset(
        #     "hotpot_qa", "fullwiki", split="train", trust_remote_code=True
        # )
        # hf_official_dev = load_dataset(
        #     "hotpot_qa", "fullwiki", split="validation", trust_remote_code=True
        # )
        hf_official_train = load_dataset(
            "hotpot_qa", "fullwiki", split="train"
        )
        hf_official_dev = load_dataset(
            "hotpot_qa", "fullwiki", split="validation"
        )
        data_path_dir = os.path.dirname(data_path)
        # save all the original data
        all_original_keys = hf_official_train[0].keys()
        for split, examples in zip(
            ["hf_official_train", "hf_official_dev"],
            [hf_official_train, hf_official_dev],
        ):
            target_path = os.path.join(data_path_dir, f"{split}.csv")
            save_csv(examples, f=target_path, fieldnames=all_original_keys)
            # for example in examples:
            #     # is answer in the context
            #     print(f"example: {example}")
            #     context = str(json.dumps(example["context"]))
            #     if example["answer"] in context:
            #         print(f"answer in context")
            #     else:
            #         print(f"answer not in context")
            print(f"saved {split} to {target_path}")
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
            keys = ["id", "question", "answer", "supporting_facts", "context"]

        official_train = []  # 15661
        for raw_example in hf_official_train:
            if raw_example["level"] == "hard":
                example = {k: raw_example[k] for k in keys}

                if "supporting_facts" in example:
                    example["gold_titles"] = set(example["supporting_facts"]["title"])
                    # del example["supporting_facts"]

                official_train.append(example)
        print(f"official_train: {len(official_train)}")

        rng = random.Random(0)
        rng.shuffle(official_train)

        sampled_trainset = official_train[: len(official_train) * 70 // 100]  # 11745
        print(f"sampled_trainset: {len(sampled_trainset)}")

        sampled_valset = official_train[  # 3916
            len(official_train) * 70 // 100 :
        ]  # this is not the official dev set

        print(f"sampled_valset: {len(sampled_valset)}")

        # for example in self._train:
        #     if keep_details == "dev_titles":
        #         del example["gold_titles"]

        test = []  # 7405

        print(f"raw_example: {hf_official_dev[0]}")
        for raw_example in hf_official_dev:
            assert raw_example["level"] == "hard"
            example = {
                k: raw_example[k]
                for k in ["id", "question", "answer", "type", "supporting_facts"]
            }
            if "supporting_facts" in example:
                example["gold_titles"] = set(example["supporting_facts"]["title"])

                # del example["supporting_facts"]
            test.append(example)

        keys = ["id", "question", "answer", "gold_titles", "context"]

        # split test into val and test
        # random shuff the test
        rng.shuffle(test)
        test_split = test[: len(test) * 50 // 100]  # 3702
        val_split = test[len(test) * 50 // 100 :]  # 3703

        # save to csv
        for split, examples in zip(
            ["train", "val", "test"],
            [sampled_trainset, val_split, test_split],
        ):
            # target_path = prepare_dataset_path(self.root, task_name, split)
            target_path = os.path.join(data_path_dir, f"{split}.json")
            # filter the examples with only the keys
            save_examples: List[HotPotQAData] = []
            for example in examples:
                save_example = {k: example[k] for k in keys if k in example}
                save_example = HotPotQAData.from_dict(save_example)
                save_examples.append(save_example.to_dict())
            save_json(save_examples, f=target_path)
            if split == "train":
                print(f"train example: {examples[0]}")
            print(f"saved {split} to {target_path}")

        if split == "train":
            return sampled_trainset
        elif split == "val":
            return sampled_valset
        elif split == "test":
            return test
        else:
            raise ValueError("Split must be one of 'train', 'val', 'test'")

    def __getitem__(self, index) -> DataClass:
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = HotPotQA(split="train")
    print(dataset[0], type(dataset[0]))
    print(len(dataset))
    valdataset = HotPotQA(split="val")
    print(len(valdataset))
    testdataset = HotPotQA(split="test")
    print(len(testdataset))

    # example = {
    #     "id": "5a8b57f25542995d1e6f1371",
    #     "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    #     "answer": "yes",
    #     "type": "comparison",
    #     "level": "hard",
    #     "supporting_facts": {
    #         "title": ["Scott Derrickson", "Ed Wood"],
    #         "sent_id": [0, 0],
    #     },
    #     "context": {
    #         "title": [
    #             "Adam Collis",
    #             "Ed Wood (film)",
    #             "Tyler Bates",
    #             "Doctor Strange (2016 film)",
    #             "Hellraiser: Inferno",
    #             "Sinister (film)",
    #             "Deliver Us from Evil (2014 film)",
    #             "Woodson, Arkansas",
    #             "Conrad Brooks",
    #             "The Exorcism of Emily Rose",
    #         ],
    #         "sentences": [
    #             [
    #                 "Adam Collis is an American filmmaker and actor.",
    #                 " He attended the Duke University from 1986 to 1990 and the University of California, Los Angeles from 2007 to 2010.",
    #                 " He also studied cinema at the University of Southern California from 1991 to 1997.",
    #                 ' Collis first work was the assistant director for the Scott Derrickson\'s short "Love in the Ruins" (1995).',
    #                 ' In 1998, he played "Crankshaft" in Eric Koyanagi\'s "Hundred Percent".',
    #             ],
    #             [
    #                 "Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood.",
    #                 " The film concerns the period in Wood's life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau.",
    #                 " Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast.",
    #             ],
    #             [
    #                 "Tyler Bates (born June 5, 1965) is an American musician, music producer, and composer for films, television, and video games.",
    #                 ' Much of his work is in the action and horror film genres, with films like "Dawn of the Dead, 300, Sucker Punch," and "John Wick."',
    #                 " He has collaborated with directors like Zack Snyder, Rob Zombie, Neil Marshall, William Friedkin, Scott Derrickson, and James Gunn.",
    #                 ' With Gunn, he has scored every one of the director\'s films; including "Guardians of the Galaxy", which became one of the highest grossing domestic movies of 2014, and its 2017 sequel.',
    #                 ' In addition, he is also the lead guitarist of the American rock band Marilyn Manson, and produced its albums "The Pale Emperor" and "Heaven Upside Down".',
    #             ],
    #             [
    #                 "Doctor Strange is a 2016 American superhero film based on the Marvel Comics character of the same name, produced by Marvel Studios and distributed by Walt Disney Studios Motion Pictures.",
    #                 " It is the fourteenth film of the Marvel Cinematic Universe (MCU).",
    #                 " The film was directed by Scott Derrickson, who wrote it with Jon Spaihts and C. Robert Cargill, and stars Benedict Cumberbatch as Stephen Strange, along with Chiwetel Ejiofor, Rachel McAdams, Benedict Wong, Michael Stuhlbarg, Benjamin Bratt, Scott Adkins, Mads Mikkelsen, and Tilda Swinton.",
    #                 ' In "Doctor Strange", surgeon Strange learns the mystic arts after a career-ending car accident.',
    #             ],
    #             [
    #                 "Hellraiser: Inferno (also known as Hellraiser V: Inferno) is a 2000 American horror film.",
    #                 ' It is the fifth installment in the "Hellraiser" series and the first "Hellraiser" film to go straight-to-DVD.',
    #                 " It was directed by Scott Derrickson and released on October 3, 2000.",
    #                 " The film concerns a corrupt detective who discovers Lemarchand's box at a crime scene.",
    #                 " The film's reviews were mixed.",
    #             ],
    #             [
    #                 "Sinister is a 2012 supernatural horror film directed by Scott Derrickson and written by Derrickson and C. Robert Cargill.",
    #                 " It stars Ethan Hawke as fictional true-crime writer Ellison Oswalt who discovers a box of home movies in his attic that puts his family in danger.",
    #             ],
    #             [
    #                 "Deliver Us from Evil is a 2014 American supernatural horror film directed by Scott Derrickson and produced by Jerry Bruckheimer.",
    #                 ' The film is officially based on a 2001 non-fiction book entitled "Beware the Night" by Ralph Sarchie and Lisa Collier Cool, and its marketing campaign highlighted that it was "inspired by actual accounts".',
    #                 " The film stars Eric Bana, Édgar Ramírez, Sean Harris, Olivia Munn, and Joel McHale in the main roles and was released on July 2, 2014.",
    #             ],
    #             [
    #                 "Woodson is a census-designated place (CDP) in Pulaski County, Arkansas, in the United States.",
    #                 " Its population was 403 at the 2010 census.",
    #                 " It is part of the Little Rock–North Little Rock–Conway Metropolitan Statistical Area.",
    #                 " Woodson and its accompanying Woodson Lake and Wood Hollow are the namesake for Ed Wood Sr., a prominent plantation owner, trader, and businessman at the turn of the 20th century.",
    #                 " Woodson is adjacent to the Wood Plantation, the largest of the plantations own by Ed Wood Sr.",
    #             ],
    #             [
    #                 "Conrad Brooks (born Conrad Biedrzycki on January 3, 1931 in Baltimore, Maryland) is an American actor.",
    #                 " He moved to Hollywood, California in 1948 to pursue a career in acting.",
    #                 ' He got his start in movies appearing in Ed Wood films such as "Plan 9 from Outer Space", "Glen or Glenda", and "Jail Bait."',
    #                 " He took a break from acting during the 1960s and 1970s but due to the ongoing interest in the films of Ed Wood, he reemerged in the 1980s and has become a prolific actor.",
    #                 " He also has since gone on to write, produce and direct several films.",
    #             ],
    #             [
    #                 "The Exorcism of Emily Rose is a 2005 American legal drama horror film directed by Scott Derrickson and starring Laura Linney and Tom Wilkinson.",
    #                 " The film is loosely based on the story of Anneliese Michel and follows a self-proclaimed agnostic who acts as defense counsel (Linney) representing a parish priest (Wilkinson), accused by the state of negligent homicide after he performed an exorcism.",
    #             ],
    #         ],
    #     },
    # }

    # # save to csv
    # keys = ["id", "question", "answer", "gold_titles", "context"]
    # example["gold_titles"] = set(example["supporting_facts"]["title"])

    # # test, save to hotpotQA

    # data = HotPotQAData.from_dict({k: example[k] for k in keys})
    # print(f"data: {data}")

    # # save to json
    # save_json([data.to_dict()], f="test.json")

    # # load from json
    # loaded_data = load_json("test.json")
    # # convert to dataclass
    # data = HotPotQAData.from_dict(loaded_data[0])
    # print(f"data: {data}")
