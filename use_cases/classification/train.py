from typing import Any, Dict, Tuple
from utils import save, load
from torch.utils.data import DataLoader


from optim.optimizer import BootstrapFewShot
from optim.sampler import RandomSampler, ClassSampler
from use_cases.classification.task import TRECClassifier
from use_cases.classification.eval import ClassifierEvaluator
from core.component import Component
from use_cases.classification.data import (
    SamplesToStr,
    load_datasets,
    _COARSE_LABELS_DESC,
    _COARSE_LABELS,
)


class Orchestrator(Component):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._example_input = "What is the capital of France?"

    # @property
    # def example_input(self):
    #     return "How did serfdom develop in and then leave Russia ?"

    # @example_input.setter
    # def example_input(self, value):
    #     self._example_input = value

    # def training_step(self, *args, **kwargs) -> None:
    #     raise NotImplementedError("training_step method is not implemented")

    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError("train method is not implemented")

    def _extra_repr(self) -> str:
        return super()._extra_repr() + f"example_input={self._example_input}"


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
        batch_size: int = 6,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.task = TRECClassifier(
            labels=_COARSE_LABELS, labels_desc=_COARSE_LABELS_DESC
        )

        self.example_input = "How did serfdom develop in and then leave Russia ?"
        self.num_shots = num_shots
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset

        self.data_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

        self.evaluator = ClassifierEvaluator(num_classes=self.num_classes)
        self.samples_to_str = SamplesToStr()

        self.params = dict(self.task.named_parameters())
        print(f"params: {self.params}")

        self.sampler = RandomSampler(
            dataset=self.train_dataset, default_num_shots=self.num_shots
        )
        self.class_sampler = ClassSampler(
            self.train_dataset,
            self.num_classes,
            get_data_key_fun=lambda x: x["coarse_label"],
        )

        self.few_shot_optimizer = BootstrapFewShot(
            parameter=self.params["generator.examples_str"],
            sampler=self.class_sampler,
            output_processors=self.samples_to_str,
            num_shots=self.num_shots,
        )

        print(f"few_shot_optimizer: {self.few_shot_optimizer}")
        print(f"few_shot_state_dict: {self.few_shot_optimizer.state_dict()}")

    def eval(self, dataset=None) -> Tuple[float, float]:
        r"""
        TODO: automatically tracking the average inference time
        """
        responses = []
        targets = []
        num_invalid = 0
        if dataset is None:
            dataset = self.eval_dataset
        for data in dataset:
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
        weights_per_class = self.evaluator.weights_per_class(responses, targets)
        return accuracy, macro_f1_score, weights_per_class

    def test(self):
        return self.eval(self.test_dataset)

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
        max_steps = 20
        # self.few_shot_optimizer.init()
        self.task.train()
        save(
            self.task.state_dict(),
            f"use_cases/classification/checkpoints/task_start",
        )
        acc, macro_f1, best_weights_per_class = self.eval()  # zero shot, 0.542
        best_score = acc + macro_f1
        print(
            f"Eval Accuracy Zero shot Start: {acc}, F1: {macro_f1}, score: {best_score}, best_weights_per_class: {best_weights_per_class}"
        )
        acc_test, macro_f1_test, weights_per_class_test = self.test()
        print(
            f"Test Accuracy Zero shot Start: {acc_test}, F1: {macro_f1_test}, score: {best_score}, weights_per_class: {weights_per_class_test}"
        )
        # compute weights per data point in training set
        weights = [
            best_weights_per_class[int(sample["coarse_label"])]
            for sample in self.train_dataset
        ]

        self.few_shot_optimizer.init(weights=weights)

        acc, macro_f1, best_weights_per_class = self.eval()  # 6 shots, class_balanced

        print(
            f"Eval Accuracy Start: {acc}, F1: {macro_f1}, score: {best_score}, best_weights_per_class: {best_weights_per_class}"
        )

        acc_test, macro_f1_test, weights_per_class_test = self.test()
        print(
            f"Test Accuracy Start: {acc_test}, F1: {macro_f1_test}, score: {best_score}, weights_per_class: {weights_per_class_test}"
        )
        start_shots = shots

        # this simulates the gradients, which will decrease the more steps we take
        # the samples to replace are weighted by the class weights
        def get_replace_shots(
            start_shot: int,
            end_shot: int = 1,
            max_step=3,
            current_step=0,
        ):
            # the number of thots will decrease from start_shot to end_shot
            gradient = float(start_shot - end_shot) / max_step
            value = int(start_shot - gradient * current_step)
            value = min(value, start_shot)
            value = max(value, end_shot)

            return value

        for i, train_batch in enumerate(self.data_loader):
            save(
                self.task.state_dict(),
                f"use_cases/classification/checkpoints/task_{i}",
            )

            if i >= max_steps:

                break
            print(f"step: {i}")
            print(f"train_batch: {train_batch}")
            replace_shots = get_replace_shots(
                start_shots, end_shot=1, max_step=max_steps, current_step=i
            )

            self.few_shot_optimizer.propose(
                shots=replace_shots, weights_per_class=best_weights_per_class
            )  # random replace half of samples

            acc1, macro_f1_1, weights_per_class = (
                self.eval()
            )  # self.batch_eval(train_batch)

            score_1 = acc1 + macro_f1_1
            print(
                f"Eval Accuracy {i} proposed: {acc1}, F1: {macro_f1_1}, score: {score_1}"
            )

            # break
            if score_1 > best_score:
                best_score = score_1
                best_weights_per_class = weights_per_class
                # update the value
                # self.few_shot_optimizer.update_parameter()
                best_parameters = self.task.state_dict()
                self.few_shot_optimizer.update_parameter()
                print(f"best_score: {best_score}")
                print(f"best_parameters: {best_parameters}")
                print(f"best_weights_per_class: {best_weights_per_class}")
            else:
                self.few_shot_optimizer.reset_parameter()
                print(f"reset_parameter")

        # # final evaluation
        acc, macro_f1, weights_per_class = self.test()
        print(
            f"Test Accuracy: {acc}, F1: {macro_f1}, weights_per_class: {weights_per_class}"
        )
        print(f"best_score: {best_score}")


if __name__ == "__main__":
    import logging
    import sys

    # Configure logging to output to standard output (console)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    # Example of setting logging to debug level
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    train_dataset, eval_dataset, test_dataset = load_datasets()
    # TODO: ensure each time the selected eval and test dataset and train dataset are the same
    num_shots = 6
    batch_size = 10
    trainer = TrecTrainer(
        num_classes=6,
        train_dataset=train_dataset,  # use for few-shot sampling
        eval_dataset=eval_dataset,  # evaluting during icl
        test_dataset=test_dataset,  # the final testing
        num_shots=num_shots,
        batch_size=batch_size,
    )
    trainer.train(shots=num_shots)
