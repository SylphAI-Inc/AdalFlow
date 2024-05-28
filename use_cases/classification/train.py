from use_cases.classification.task import TRECClassifier
from use_cases.classification.eval import ClassifierEvaluator
from core.component import Component
from use_cases.classification.data import (
    SamplesToStr,
    dataset,
    _COARSE_LABELS_DESC,
    _COARSE_LABELS,
)
from torch.utils.data import DataLoader
import random


from typing import Any, Optional, Sequence, Dict
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, RandomSampler
from utils import save, load


class Orchestrator(Component):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._example_input = "What is the capital of France?"

    @property
    def example_input(self):
        return "How did serfdom develop in and then leave Russia ?"

    @example_input.setter
    def example_input(self, value):
        self._example_input = value

    # def training_step(self, *args, **kwargs) -> None:
    #     raise NotImplementedError("training_step method is not implemented")

    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError("train method is not implemented")

    def _extra_repr(self) -> str:
        return super()._extra_repr() + f"example_input={self._example_input}"


from optimizer.optimizer import BootstrapFewShot
from optimizer.sampler import RandomSampler, ClassSampler
from typing import Tuple


# for this trainer, we will learn from pytorch lightning
class TrecTrainer(Orchestrator):
    r"""
    data loader which is random shuffed already, and the batch can be used as the # samples
    """

    def __init__(
        self,
        num_classes: int,
        train_dataset,
        eval_dataset,
        test_dataset=None,
        num_shots: int = 5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.task = TRECClassifier(
            labels=_COARSE_LABELS, labels_desc=_COARSE_LABELS_DESC
        )

        self.example_input = "How did serfdom develop in and then leave Russia ?"
        self.num_shots = 8
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.data_loader = DataLoader(
            self.train_dataset, batch_size=self.num_shots, shuffle=True
        )
        self.eval_data_loader = DataLoader(
            self.eval_dataset, batch_size=1, shuffle=False
        )  # use this for speeded up evaluation
        self.evaluator = ClassifierEvaluator(num_classes=self.num_classes)
        self.samples_to_str = SamplesToStr()

        self.params = dict(self.task.named_parameters())
        print(f"params: {self.params}")

        self.sampler = RandomSampler(
            dataset=self.train_dataset, num_shots=self.num_shots
        )
        self.class_sampler = ClassSampler(
            self.train_dataset,
            self.num_classes,
            get_data_key_fun=lambda x: x["coarse_label"],
        )
        self.few_shot_optimizer = BootstrapFewShot(
            parameter_dict=self.params,
            parameter_name="generator.examples_str",
            # parameter_names=["generator.examples_str"],
            # train_dataset=self.train_dataset,
            sampler=self.class_sampler,
            output_processors=self.samples_to_str,
        )

        print(f"few_shot_optimizer: {self.few_shot_optimizer}")
        print(f"few_shot_state_dict: {self.few_shot_optimizer.state_dict()}")

    def eval(self):
        r"""
        TODO: automatically tracking the average inference time
        """
        responses = []
        targets = []
        num_invalid = 0
        for data in self.eval_dataset.select(range(20)):
            print(f"data: {data}")
            task_input = data["text"]
            corse_label = data["coarse_label"]
            print(f"task_input: {task_input}, corse_label: {corse_label}")
            print(f"types: {type(task_input)}, {type(corse_label)}")

            response = self.task(task_input)
            if response == -1:
                print(f"invalid response: {response}")
                num_invalid += 1
                continue
            responses.append(response)
            targets.append(int(corse_label))

        # evaluate the responses
        print(f"responses: {responses}, targets: {targets}")
        print(f"num_invalid: {num_invalid}")
        accuracy, macro_f1_score = self.evaluator.run(responses, targets)
        return accuracy, macro_f1_score

    def batch_eval(self, batch: Dict[str, Any]) -> Tuple[float, float]:
        r"""
        batch evaluation
        """
        responses = []
        targets = []
        num_invalid = 0
        for text, corse_label in zip(batch["text"], batch["coarse_label"]):
            # print(f"data: {data}")
            task_input = text
            # corse_label = data["coarse_label"]
            print(f"task_input: {task_input}, corse_label: {corse_label}")
            print(f"types: {type(task_input)}, {type(corse_label)}")

            response = self.task(task_input)
            if response == -1:
                print(f"invalid response: {response}")
                num_invalid += 1
                continue
            responses.append(response)
            targets.append(int(corse_label))

        # evaluate the responses
        print(f"responses: {responses}, targets: {targets}")
        print(f"num_invalid: {num_invalid}")
        accuracy, macro_f1_score = self.evaluator.run(responses, targets)
        return accuracy, macro_f1_score

    def train(self, shots: int) -> None:
        r"""
        ICL with demonstrating examples, we might want to know the plot of the accuracy while using the few shots examples
        """

        best_parameters = None
        best_eval = None
        best_score = 0
        max_steps = 4
        # for step in range(max_steps):
        for i, train_batch in enumerate(self.data_loader):
            if i >= max_steps:
                break
            print(f"step: {i}")
            print(f"train_batch: {train_batch}")
            self.few_shot_optimizer.step(num_shots=shots)
            states = self.task.state_dict()
            save(
                self.task.state_dict(),
                f"use_cases/classification/checkpoints/task_{i}",
            )
            # try load from the file
            json_states, pickle_states = load(
                f"use_cases/classification/checkpoints/task_{i}"
            )
            # pickle state is the same as states
            self.task.load_state_dict(pickle_states)
            acc, macro_f1 = self.batch_eval(train_batch)  # should do batch evaluation
            print(f"Eval Accuracy: {acc}, F1: {macro_f1}")
            score = acc + macro_f1
            # break
            if score > best_score:
                best_score = score
                best_eval = (acc, macro_f1)
                best_parameters = states
                print(f"best_score: {best_score}")
        print(f"best_parameters: {best_parameters}")
        print(f"best_eval: {best_eval}")

        # final evaluation
        acc, macro_f1 = self.eval()
        print(f"Eval Accuracy: {acc}, F1: {macro_f1}")

        # samples_str = random_class_balanced_str_with_thought_space
        # print(f"samples_str: {samples_str}")
        # # return
        # state_dict = {
        #     "generator": {"preset_prompt_kwargs": {"examples_str": samples_str}}
        # }
        # self.task.load_state_dict(state_dict)
        # self.task.generator.print_prompt()
        # acc, macro_f1 = self.eval()
        print(f"Eval Accuracy: {acc}, F1: {macro_f1}")


if __name__ == "__main__":
    import logging
    import sys

    # Configure logging to output to standard output (console)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    # Example of setting logging to debug level
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    trainer = TrecTrainer(
        num_classes=6, train_dataset=train_dataset, eval_dataset=eval_dataset
    )
    # # print(trainer)
    trainer.train(6)
