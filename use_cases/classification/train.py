from use_cases.classification.task import TRECClassifier
from use_cases.classification.eval import ClassifierEvaluator
from core.component import Component
from use_cases.classification.data import (
    TrecDataset,
    ToSampleStr,
    dataset,
    _COARSE_LABELS_DESC,
    _COARSE_LABELS,
    _FINE_LABELS,
)
from torch.utils.data import DataLoader
import random


from typing import Any, Optional, Sequence, Dict
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, RandomSampler


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


class ExampleOptimizer(Component):
    def __init__(self, dataset, num_shots: int) -> None:
        super().__init__()
        self.dataset = dataset
        # self.sampler = RandomSampler(self.dataset, replacement=False, num_samples=num_shots)

    def random_sample(self, shots: int, dataset) -> Sequence[str]:
        indices = random.sample(range(len(dataset)), shots)
        return [self.dataset[i] for i in indices]

        # sampler = RandomSampler(self.dataset, replacement=False, num_samples=shots)
        # iter_times = shots // len(self.dataset)
        # samples = []
        # # use sampler.next() to get the next sample
        # for i in range(iter_times):
        #     samples.append(self.sampler.next())


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
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.task = TRECClassifier(
            labels=_COARSE_LABELS, labels_desc=_COARSE_LABELS_DESC
        )
        self.example_input = "How did serfdom develop in and then leave Russia ?"
        self.default_num_shots = 5
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.data_loader = DataLoader(
            self.train_dataset, batch_size=self.default_num_shots, shuffle=True
        )
        self.eval_data_loader = DataLoader(
            self.eval_dataset, batch_size=1, shuffle=False
        )  # use this for speeded up evaluation
        self.sample_optimizer = ExampleOptimizer(
            self.train_dataset, self.default_num_shots
        )
        self.evaluator = ClassifierEvaluator(num_classes=self.num_classes)
        self.to_sample_str = ToSampleStr()

    def random_shots(self, shots: int) -> Sequence[str]:
        samples = self.sample_optimizer.random_sample(shots)
        samples_str = [self.to_sample_str(sample) for sample in samples]
        return samples_str

    def eval(self):
        responses = []
        targets = []
        num_invalid = 0
        for data in self.eval_dataset.select(range(10)):
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

    def train(self, shots: int) -> None:
        r"""
        ICL with demonstrating examples, we might want to know the plot of the accuracy while using the few shots examples
        """
        samples = self.sample_optimizer.random_sample(shots, self.train_dataset)
        samples_str = [self.to_sample_str(sample) for sample in samples]
        # state_dict = {
        #     "generator": {"preset_prompt_kwargs": {"examples_str": samples_str}}
        # }
        # self.task.load_state_dict(state_dict)
        self.task.generator.print_prompt()
        acc, macro_f1 = self.eval()
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
    print(trainer)
    trainer.train(20)
