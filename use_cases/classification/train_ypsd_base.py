from use_cases.classification.eval import ClassifierEvaluator
from core.component import Component
from use_cases.classification.data import (
    dataset,
    extract_class_label,
)
import random

from use_cases.classification.task_ypsd import BaseGenerator


from typing import Sequence


# for this trainer, we will learn from pytorch lightning


from dspy.teleprompt import BootstrapFewShot
from use_cases.classification.task_ypsd import CoT
from typing import Callable

# config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)


def parse_output(pred):
    output = pred.class_index if pred and pred.class_index else -1
    parsed_output = extract_class_label(output)
    return parsed_output


def eval_a_task(task: Callable, eval_dataset, evaluator):
    responses = []
    targets = []
    num_invalid = 0
    for data in eval_dataset.select(range(0, 20)):
        task_input = data["text"]
        coarse_label = data["coarse_label"]
        response = task(task_input)
        print(f"task_input: {task_input}, coarse_label: {coarse_label}")
        print(f"response: {response}")
        if response == -1:
            num_invalid += 1
            continue
        response = parse_output(response)
        responses.append(response)
        targets.append(int(coarse_label))
    print(f"responses: {responses}, targets: {targets}")
    print(f"num_invalid: {num_invalid}")
    accuracy, macro_f1_score = evaluator.run(responses, targets)
    return accuracy, macro_f1_score


# dspy caching is error-prone, as the structure can change. The signature.
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric, parse_integer_answer
from use_cases.classification.data import dataset as HFDataset

# def acc_metric(gt, pred):

# gsm8k = GSM8K()

import random

import tqdm
from datasets import load_dataset


# def acc_metric(gold, pred, trace=None):
#     print(f"gold: {gold}, pred: {pred}")
#     return int(parse_integer_answer(str(gold.answer))) == int(int(pred))


def acc_metric(gold, pred, trace=None):
    print(f"gold: {gold}, pred: {pred}")
    return int(parse_integer_answer(str(gold.answer))) == int(
        parse_integer_answer(str(pred.class_index))
    )


class TREC5k:
    def __init__(self) -> None:
        super().__init__()
        self.do_shuffle = False

        dataset = HFDataset
        hf_official_train = dataset["train"]
        hf_official_test = dataset["test"]
        official_train = []
        official_test = []

        for example in tqdm.tqdm(hf_official_train):
            question = example["text"]

            answer = example["coarse_label"]

            # gold_reasoning = " ".join(answer[:-2])
            # answer = str(int(answer[-1].replace(",", "")))
            gold_reasoning = None

            official_train.append(
                dict(question=question, gold_reasoning=gold_reasoning, answer=answer)
            )

        for example in tqdm.tqdm(hf_official_test):
            question = example["text"]

            answer = example["coarse_label"]

            # gold_reasoning = " ".join(answer[:-2])
            # answer = str(int(answer[-1].replace(",", "")))
            gold_reasoning = None

            official_test.append(
                dict(question=question, gold_reasoning=gold_reasoning, answer=answer)
            )

        rng = random.Random(0)
        rng.shuffle(official_train)

        rng = random.Random(0)
        rng.shuffle(official_test)

        trainset = official_train[:200]
        devset = official_train[200:500]
        testset = official_test[:]

        import dspy

        trainset = [dspy.Example(**x).with_inputs("question") for x in trainset]
        devset = [dspy.Example(**x).with_inputs("question") for x in devset]
        testset = [dspy.Example(**x).with_inputs("question") for x in testset]

        # print(f"Trainset size: {len(trainset)}")
        # print(f"Devset size: {len(devset)}")
        # print(f"Testset size: {len(testset)}")

        self.train = trainset
        self.dev = devset
        self.test = testset


class Trainer:
    def __init__(self, train_dataset, eval_dataset):
        self.task = BaseGenerator()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.evaluator = ClassifierEvaluator(num_classes=6)
        self.train_example_set = TREC5k().train[0:10]

    def eval_baseline(self, task):
        acc = eval_a_task(task, self.eval_dataset, self.evaluator)
        print(f"acc: {acc}")
        self.task.save("use_cases/classification/task_dspy.txt")
        return acc

    def train(self):

        # Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
        config = dict(
            max_bootstrapped_demos=5, max_labeled_demos=5  # , num_candidate_programs=2
        )
        # Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
        # (0.9, 0.856)
        teleprompter = BootstrapFewShot(metric=acc_metric, **config)

        optimized_cot = teleprompter.compile(self.task, trainset=self.train_example_set)
        print(optimized_cot)
        # get current path
        import os

        path = os.getcwd() + "/use_cases/classification/optimized_cot.txt"
        optimized_cot.save(path)
        metrics = self.eval_baseline(optimized_cot)
        print(metrics)
        return metrics
        # print(f"optimized_cot state: {optimized_cot.dump_state()}")


if __name__ == "__main__":
    import logging
    import sys

    # 1 no thought base: (0.55, 0.531)
    # 2 with bootstrapFewShot: (0.75, 0.708)
    # 3 CoT with bootstrapFewShot: (0.9, 0.856)

    # Configure logging to output to standard output (console)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    # Example of setting logging to debug level
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    trainer = Trainer(train_dataset=train_dataset, eval_dataset=eval_dataset)
    # metrics = trainer.eval_baseline(trainer.task)
    trainer.train()
    # print(metrics)
