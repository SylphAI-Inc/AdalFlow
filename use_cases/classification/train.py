from typing import Any, Dict, Tuple, List
import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader

from lightrag.components.model_client import (
    GroqAPIClient,
    OpenAIClient,
    GoogleGenAIClient,
)
from lightrag.optim import BootstrapFewShot
from lightrag.optim.sampler import RandomSampler, ClassSampler
from lightrag.optim.llm_augment import LLMAugmenter
from lightrag.optim.llm_optimizer import LLMOptimizer

from lightrag.core.component import Component
from lightrag.utils import save, load, save_json, load_json, save_json_from_dict


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

        # self.example_input = "How did serfdom develop in and then leave Russia ?"
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

        # OR use dataloader
        print(f"dataset: {dataset}")
        subset = dataset.select(range(0, 10))
        for text, coarse_label in tqdm.tqdm(
            zip(subset["text"], subset["coarse_label"])
        ):
            log.info(f"data: text: {text}, coarse_label: {coarse_label}")
            # task_input = data["text"]
            # corse_label = data["coarse_label"]
            # print(f"task_input: {task_input}, corse_label: {corse_label}")
            # print(f"types: {type(task_input)}, {type(corse_label)}")

            response = self.task(text)
            if response == -1:
                log.error(f"invalid response: {response}")
                num_invalid += 1
                continue
            responses.append(response)
            targets.append(int(coarse_label))

        # evaluate the responses
        log.info(f"responses: {responses}, targets: {targets}")
        log.info(f"num_invalid: {num_invalid}")
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

    def eval_zero_shot(self, save_path: str = None):
        save_path = save_path or "use_cases/classification/evals/zero_shot.json"
        json_obj: Dict[str, Any] = {}
        self.task.eval()  # not using any trained examples
        acc, macro_f1, best_weights_per_class = self.eval()  # zero shot, 0.542
        log.info(
            f"Eval Accuracy Zero shot Start: {acc}, F1: {macro_f1}, score: {acc+macro_f1}, best_weights_per_class: {best_weights_per_class}"
        )
        acc_test, macro_f1_test, weights_per_class_test = self.test()
        log.info(
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
        save_json(json_obj, save_path)

    def eval_few_shot(self, shots: int, runs: int = 5, save_path: str = None):
        r"""Get the max, min, mean, std of the few shot evaluation"""
        # TODO: this can be moved to the optimizer
        save_path = save_path or "use_cases/classification/evals/few_shot.json"

        def compute_max_min_mean_std(values: List[float]):
            import numpy as np

            values_np = np.array(values)
            max_value = np.max(values_np)
            min_value = np.min(values_np)
            mean_value = np.mean(values_np)
            std_value = np.std(values_np)
            return max_value, min_value, mean_value, std_value

        self.task.train()
        accs = []
        macro_f1s = []

        accs_eval = []
        macro_f1s_eval = []
        optimizer = self.few_shot_optimizer

        # get optimizer name
        optimizer_name = (
            optimizer.__class__.__name__ + optimizer.sampler.__class__.__name__
        )
        result: Dict[str, Any] = {
            "optimizer": optimizer_name,
            "shots": shots,
            "runs": runs,
        }
        if shots is None:
            shots = self.num_shots
        for i in tqdm.tqdm(range(runs)):
            optimizer.init(shots=shots)
            log.info(f"run: {i}, eval")
            acc_eval, macro_f1_eval, _ = self.eval()
            log.info(f"run: {i}, test")
            acc, macro_f1, _ = self.test()
            accs.append(acc)
            macro_f1s.append(macro_f1)
            accs_eval.append(acc_eval)
            macro_f1s_eval.append(macro_f1_eval)
            result[f"run_test_{i}"] = {
                "acc": acc,
                "macro_f1": macro_f1,
                "examples": optimizer.current,
            }
            result[f"run_eval_{i}"] = {
                "acc": acc_eval,
                "macro_f1": macro_f1_eval,
                "examples": optimizer.current,
            }
            log.info(result[f"run_test_{i}"])
            log.info(result[f"run_eval_{i}"])

        max_acc, min_acc, mean_acc, std_acc = compute_max_min_mean_std(accs)
        max_acc_eval, min_acc_eval, mean_acc_eval, std_acc_eval = (
            compute_max_min_mean_std(accs_eval)
        )
        log.info(
            f"test: max_acc: {max_acc}, min_acc: {min_acc}, mean_acc: {mean_acc}, std_acc: {std_acc}"
        )
        log.info(
            f"eval: max_acc: {max_acc_eval}, min_acc: {min_acc_eval}, mean_acc: {mean_acc_eval}, std_acc: {std_acc_eval}"
        )

        result["test_acc"] = {
            "max_acc": max_acc,
            "min_acc": min_acc,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
        }
        result["eval_acc"] = {
            "max_acc": max_acc_eval,
            "min_acc": min_acc_eval,
            "mean_acc": mean_acc_eval,
            "std_acc": std_acc_eval,
        }

        # macro f1
        max_macro_f1, min_macro_f1, mean_macro_f1, std_macro_f1 = (
            compute_max_min_mean_std(macro_f1s)
        )
        max_macro_f1_eval, min_macro_f1_eval, mean_macro_f1_eval, std_macro_f1_eval = (
            compute_max_min_mean_std(macro_f1s_eval)
        )
        log.info(
            f"test: max_macro_f1: {max_macro_f1}, min_macro_f1: {min_macro_f1}, mean_macro_f1: {mean_macro_f1}, std_macro_f1: {std_macro_f1}"
        )
        log.info(
            f"eval: max_macro_f1: {max_macro_f1_eval}, min_macro_f1: {min_macro_f1_eval}, mean_macro_f1: {mean_macro_f1_eval}, std_macro_f1: {std_macro_f1_eval}"
        )
        result["test_macro_f1"] = {
            "max_macro_f1": max_macro_f1,
            "min_macro_f1": min_macro_f1,
            "mean_macro_f1": mean_macro_f1,
            "std_macro_f1": std_macro_f1,
        }
        result["eval_macro_f1"] = {
            "max_macro_f1": max_macro_f1_eval,
            "min_macro_f1": min_macro_f1_eval,
            "mean_macro_f1": mean_macro_f1_eval,
            "std_macro_f1": std_macro_f1_eval,
        }

        save_json(result, save_path)

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
    import sys

    from use_cases.classification.config_log import log
    from lightrag.utils import save_json

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

    # save the most detailed trainer states
    # When your dataset is small, this json file can be used to help you visualize datasets
    # and to debug components
    save_json(
        trainer.to_dict(),
        f"use_cases/classification/traces/trainer_states.json",
    )
    log.info(f"trainer to dict: {trainer.to_dict()}")
    # or log a str representation, mostly just the structure of the trainer
    log.info(f"trainer: {trainer}")
    # trainer.train_instruction(max_steps=1)
    # trainer.train(shots=num_shots, max_steps=20, start_shots=6)
    # trainer.eval_zero_shot()
    trainer.eval_few_shot(shots=num_shots, runs=5)
