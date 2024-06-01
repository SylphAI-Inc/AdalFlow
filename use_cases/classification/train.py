from typing import Any, Dict, Tuple
from utils import save, load

from copy import deepcopy
from torch.utils.data import DataLoader

from components.api_client import GroqAPIClient, OpenAIClient, GoogleGenAIClient
from optim.optimizer import BootstrapFewShot
from optim.sampler import RandomSampler, ClassSampler
from optim.llm_augment import LLMAugmenter
from optim.llm_optimizer import LLMOptimizer

from core.component import Component

from use_cases.classification.task import TRECClassifier, InputFormat, OutputFormat
from use_cases.classification.eval import ClassifierEvaluator
from use_cases.classification.data import (
    SamplesToStr,
    load_datasets,
    _COARSE_LABELS_DESC,
    _COARSE_LABELS,
)


# for this trainer, we will learn from pytorch lightning
class TrecTrainer(Component):
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

        model_client = self.task.generator.model_client
        model_client = OpenAIClient()

        model_kwargs = deepcopy(self.task.generator.model_kwargs)
        model_kwargs = {"model": "gpt-4o"}
        print(f"model_client: {model_client}")
        print(f"model_kwargs: {model_kwargs}")
        task_context_str = self.task.task_desc_str

        # creating task template
        self.example_augmenter = LLMAugmenter(
            model_client=model_client,
            model_kwargs=model_kwargs,
            task_context_str=task_context_str,
        )

        self.evaluator = ClassifierEvaluator(num_classes=self.num_classes)
        self.samples_to_str = SamplesToStr()

        self.params = dict(self.task.named_parameters())
        print(f"params: {self.params}")

        self.sampler = RandomSampler[Dict](
            dataset=self.train_dataset, default_num_shots=self.num_shots
        )
        self.class_sampler = ClassSampler[Dict](
            self.train_dataset,
            self.num_classes,
            get_data_key_fun=lambda x: x["coarse_label"],
        )

        self.few_shot_optimizer = BootstrapFewShot(
            parameter=self.params["generator.examples_str"],
            sampler=self.class_sampler,
            output_processors=self.samples_to_str,
            num_shots=self.num_shots,
            llm_augmenter=self.example_augmenter,
            task_input_dataclass=InputFormat,
            task_output_dataclass=OutputFormat,
        )
        self.few_shot_optimizer_random = BootstrapFewShot(
            parameter=self.params["generator.examples_str"],
            sampler=self.sampler,
            output_processors=self.samples_to_str,
            num_shots=self.num_shots,
            llm_augmenter=self.example_augmenter,
            task_input_dataclass=InputFormat,
            task_output_dataclass=OutputFormat,
        )

        print(f"few_shot_optimizer: {self.few_shot_optimizer}")
        print(
            f"few_shot_state_dict: {self.few_shot_optimizer.state_dict()}",
        )

        self.instruction_optimier = LLMOptimizer(
            self.params["generator.task_desc_str"],
            model_client=model_client,
            model_kwargs=model_kwargs,
        )

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
        # print(f"test_dataset", self.test_dataset)
        # sub_test_dataset = self.test_dataset.select(range(0, 1))
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

    def eval_zero_shot(self):
        json_obj: Dict[str, Any] = {}
        self.task.eval()  # not using any trained examples
        acc, macro_f1, best_weights_per_class = self.eval()  # zero shot, 0.542
        print(
            f"Eval Accuracy Zero shot Start: {acc}, F1: {macro_f1}, score: {acc+macro_f1}, best_weights_per_class: {best_weights_per_class}"
        )
        acc_test, macro_f1_test, weights_per_class_test = self.test()
        print(
            f"Test Accuracy Zero shot Start: {acc_test}, F1: {macro_f1_test}, score: {acc_test +macro_f1_test }, weights_per_class: {weights_per_class_test}"
        )
        json_obj["zero_shot"] = {
            "eval": {
                "acc": acc,
                "macro_f1": macro_f1,
            },
            "test": {
                "acc": acc_test,
                "macro_f1": macro_f1_test,
            },
        }
        save(json_obj, f"use_cases/classification/zero_shot")

    def eval_few_shot(self, shots: int, runs: int = 5):
        r"""Get the max, min, mean, std of the few shot evaluation"""
        # TODO: this can be moved to the optimizer
        self.task.train()
        accs = []
        macro_f1s = []
        optimizer = self.few_shot_optimizer

        # get optimizer name
        optimizer_name = (
            optimizer.__class__.__name__ + optimizer.sampler.__class__.__name__
        )
        save_json: Dict[str, Any] = {
            "optimizer": optimizer_name,
            "shots": shots,
            "runs": runs,
        }
        if shots is None:
            shots = self.num_shots
        for i in range(runs):  # TODO: add tqdm
            optimizer.init(shots=shots)
            acc, macro_f1, _ = self.test()
            accs.append(acc)
            macro_f1s.append(macro_f1)
            save_json[f"run_{i}"] = {
                "acc": acc,
                "macro_f1": macro_f1,
                "examples": optimizer.current,
            }
            print(save_json[f"run_{i}"])
        print(f"accs: {accs}")
        print(f"macro_f1s: {macro_f1s}")
        # compute max, min, mean, std using numpy
        import numpy as np

        accs_np = np.array(accs)
        macro_f1s_np = np.array(macro_f1s)
        max_acc = np.max(accs_np)
        min_acc = np.min(accs_np)
        mean_acc = np.mean(accs_np)
        std_acc = np.std(accs_np)
        print(
            f"max_acc: {max_acc}, min_acc: {min_acc}, mean_acc: {mean_acc}, std_acc: {std_acc}"
        )
        save_json["max_acc"] = max_acc
        save_json["min_acc"] = min_acc
        save_json["mean_acc"] = mean_acc
        save_json["std_acc"] = std_acc

        # macro f1
        max_macro_f1 = np.max(macro_f1s_np)
        min_macro_f1 = np.min(macro_f1s_np)
        mean_macro_f1 = np.mean(macro_f1s_np)
        std_macro_f1 = np.std(macro_f1s_np)
        print(
            f"max_macro_f1: {max_macro_f1}, min_macro_f1: {min_macro_f1}, mean_macro_f1: {mean_macro_f1}, std_macro_f1: {std_macro_f1}"
        )
        save_json["max_macro_f1"] = max_macro_f1
        save_json["min_macro_f1"] = min_macro_f1
        save_json["mean_macro_f1"] = mean_macro_f1
        save_json["std_macro_f1"] = std_macro_f1

        save(
            save_json,
            f"use_cases/classification/few_shot_init_1/{shots}_shots_{optimizer_name}_aug_gpt4o",
        )

    def train_random(self, shots: int) -> None:
        r"""
        ICL with random examples
        Best 0.958, 0.95
        """
        best_parameters = None
        max_steps = 5
        # self.few_shot_optimizer.init()
        self.task.train()

        self.few_shot_optimizer_random.init()
        save(
            self.task.state_dict(),
            f"use_cases/classification/checkpoints/task_start",
        )

        acc, macro_f1, best_weights_per_class = self.eval()
        best_score = acc + macro_f1
        print(f"Eval Accuracy Start: {acc}, F1: {macro_f1}, score: {best_score}")
        acc_test, macro_f1_test, weights_per_class_test = self.test()
        print(
            f"Test Accuracy Start: {acc_test}, F1: {macro_f1_test}, score: {acc_test, macro_f1_test}, weights_per_class: {weights_per_class_test}"
        )
        start_shots = 3

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
            self.few_shot_optimizer_random.propose(shots=shots)
            acc1, macro_f1_1, _ = self.eval()
            score_1 = acc1 + macro_f1_1
            print(
                f"Eval Accuracy {i} proposed: {acc1}, F1: {macro_f1_1}, score: {score_1}"
            )

            if score_1 > best_score:
                best_score = score_1
                best_parameters = self.task.state_dict()
                self.few_shot_optimizer_random.update_parameter()
                print(f"best_score: {best_score}")
                print(f"best_parameters: {best_parameters}")
                print(f"best_weights_per_class: {best_weights_per_class}")
            else:
                self.few_shot_optimizer_random.reset_parameter()
                print(f"reset_parameter")

        acc, macro_f1, weights_per_class = self.test()
        print(
            f"Test Accuracy: {acc}, F1: {macro_f1}, weights_per_class: {weights_per_class}"
        )

    def train_instruction(self, max_steps: int = 5) -> None:
        # better to provide a manual instruction
        # TODO: how to save the states.
        top_5_instructions = []
        self.task.train()
        best_score: float = 0.0
        for i, train_batch in enumerate(self.data_loader):
            if i >= max_steps:
                break

            self.instruction_optimier.propose()
            acc, f1 = self.batch_eval(train_batch)
            score = (acc + f1) / 2.0
            print(f"step: {i}")
            print(f"score: {score}")
            if score > best_score:
                best_score = score
                self.instruction_optimier.update_parameter(score)
                print(f"best_score: {best_score}")
                print(f"best_parameters: {self.params['generator.task_desc_str']}")
            else:
                self.instruction_optimier.reset_parameter()
                print(f"reset_parameter")
        # test the best instruction
        acc, macro_f1, weights_per_class = self.test()
        print(
            f"Test Accuracy: {acc}, F1: {macro_f1}, weights_per_class: {weights_per_class}"
        )
        # save the best instruction
        save(
            self.task.state_dict(),
            f"use_cases/classification/checkpoints/task_instruction/state_dict",
        )
        # save all instructions history from the optimizer
        save(
            self.instruction_optimier.instruction_history,
            f"use_cases/classification/checkpoints/task_instruction/instruction_history",
        )

    def train(self, shots: int, max_steps: int = 5, start_shots: int = 3) -> None:
        r"""
        ICL with demonstrating examples, we might want to know the plot of the accuracy while using the few shots examples
        """

        best_parameters = None
        self.task.train()

        self.few_shot_optimizer.init()
        save(
            self.task.state_dict(),
            f"use_cases/classification/checkpoints/task_start",
        )

        acc, macro_f1, best_weights_per_class = self.eval()  # 6 shots, class_balanced

        best_score = acc + macro_f1

        print(
            f"Eval Accuracy Start: {acc}, F1: {macro_f1}, score: {best_score}, best_weights_per_class: {best_weights_per_class}"
        )

        acc_test, macro_f1_test, weights_per_class_test = self.test()
        print(
            f"Test Accuracy Start: {acc_test}, F1: {macro_f1_test}, score: {acc_test, macro_f1_test}, weights_per_class: {weights_per_class_test}"
        )

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
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger()

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
    logger.info(f"trainer: {trainer}")
    # trainer.train_instruction(max_steps=1)
    # trainer.train(shots=num_shots, max_steps=20, start_shots=6)
    trainer.eval_zero_shot()
    # trainer.eval_few_shot(shots=num_shots, runs=5)
