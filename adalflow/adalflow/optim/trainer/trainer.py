"""Ready to use trainer for LLM task pipeline"""

from typing import Literal, Optional, List, Dict, Any, Tuple, TYPE_CHECKING
import os
import logging
from tqdm import tqdm
import random
import numpy as np
import uuid
import time
from copy import copy

from adalflow.core.component import Component
from adalflow.optim.optimizer import Optimizer, DemoOptimizer, TextOptimizer

if TYPE_CHECKING:
    from adalflow.optim.parameter import Parameter
    from adalflow.core.generator import BackwardPassSetup

from adalflow.optim.parameter import OutputParameter

from adalflow.optim.types import (
    PromptData,
    TrainerResult,
    ParameterType,
    TrainerStepResult,
)
from adalflow.eval.base import EvaluationResult
from adalflow.optim.trainer.adal import AdalComponent
from adalflow.optim.text_grad.ops import sum_ops

from adalflow.utils import save_json
from adalflow.utils.file_io import load_standard_json
from adalflow.utils.cache import hash_text_sha1
from adalflow.utils.data import DataLoader
from adalflow.utils.logger import printc
from adalflow.optim.types import TrainerValidateStats


logger = logging.getLogger(__name__)


class Trainer(Component):
    __doc__ = r"""Ready to use trainer for LLM task pipeline to optimize all types of parameters.


    Training set: can be used for passing initial proposed prompt or for few-shot sampling.
    Validation set: Will be used to select the final prompt or samples.
    Test set: Will be used to evaluate the final prompt or samples.

    Args:
        adaltask: AdalComponent: AdalComponent instance
        strategy: Literal["random", "constrained"]: Strategy to use for the optimizer
        max_steps: int: Maximum number of steps to run the optimizer
        num_workers: int: Number of workers to use for parallel processing
        ckpt_path: str: Path to save the checkpoint files, default to ~/.adalflow/ckpt.
        batch_val_score_threshold: Optional[float]: Threshold for skipping a batch
        max_error_samples: Optional[int]: Maximum number of error samples to keep
        max_correct_samples: Optional[int]: Maximum number of correct samples to keep
        max_proposals_per_step: int: Maximum number of proposals to generate per step
        train_loader: Any: DataLoader instance for training
        train_dataset: Any: Training dataset
        val_dataset: Any: Validation dataset
        test_dataset: Any: Test dataset
        few_shots_config: Optional[FewShotConfig]: Few shot configuration
        save_traces: bool: Save traces for for synthetic data generation or debugging
        debug: bool: Debug mode to run the trainer in debug mode. If debug is True, for text debug, the graph will be under /ckpt/YourAdalComponentName/debug_text_grads for prompt parameter,
        and for demo debug, the graph will be under /ckpt/YourAdalComponentName/debug_demos for demo parameters.

    Note:
        When you are in the debug mode, you can use get_logger api to show more detailed log on your own.

        Example:

            from adalflow.utils import get_logger

            get_logger(level="DEBUG")
    """

    adaltask: AdalComponent  # task pipeline
    train_batch_size: Optional[int] = 4

    train_loader: Any
    val_dataset = None
    test_dataset = None
    strategy: Literal["random", "constrained"]
    optimization_order: Literal["sequential", "mix"] = (
        "sequential"  # zero-shot first, bootstrap second
    )
    sequential_order: List[str] = ["text", "demo"]
    max_steps: int
    optimizer: Optimizer = None
    ckpt_path: Optional[str] = None
    ckpt_file: Optional[str] = None
    num_workers: int = 4
    max_proposals_per_step: int = 5
    # moving batch for speed up the training
    batch_val_score_threshold: Optional[float] = (
        1.0  # when acc_score >= this threshold, skip this batch
    )
    correct_val_score_threshold: Optional[float] = (
        0.5  # when acc_score >= this threshold, it is considered as correct sample
    )
    max_error_samples: Optional[int] = 2
    max_correct_samples: Optional[int] = 2
    debug: bool = False
    random_seed: int = None
    skip_subset_val: bool = False
    disable_backward_gradients: bool = False
    disable_backward: bool = False  # no backward is run at all.
    text_optimizers_config_kwargs: Optional[Dict[str, Any]] = {}

    def __init__(
        self,
        adaltask: AdalComponent,
        optimization_order: Literal["sequential", "mix"] = "sequential",
        strategy: Literal["random", "constrained"] = "constrained",  # search strategy
        max_steps: int = 1000,
        train_batch_size: Optional[int] = 4,
        num_workers: int = 4,
        ckpt_path: str = None,
        batch_val_score_threshold: Optional[float] = 1.0,
        correct_val_score_threshold: Optional[float] = 0.5,
        max_error_samples: Optional[int] = 2,
        max_correct_samples: Optional[int] = 2,
        max_proposals_per_step: int = 5,
        train_loader: Optional[Any] = None,
        train_dataset: Optional[Any] = None,
        val_dataset: Optional[Any] = None,
        test_dataset: Optional[Any] = None,
        # For demo optimizer
        raw_shots: Optional[int] = None,
        bootstrap_shots: Optional[int] = None,
        weighted_sampling: bool = False,  # if weighted sampling when do few-shot demos
        exclude_input_fields_from_bootstrap_demos: bool = False,
        debug: bool = False,
        save_traces: bool = False,  # save traces in the few-shto demos
        sequential_order: List[str] = ["text", "demo"],
        skip_subset_val: bool = False,
        disable_backward_gradients: bool = False,  # self.adaltask.disable_backward_engine controls the disable_backward engine
        disable_backward: bool = False,  # no backward is run at all.
        text_optimizers_config_kwargs: Optional[Dict[str, Any]] = {},
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        if not isinstance(adaltask, AdalComponent):
            raise ValueError("Task should be an instance of AdalComponent")
        if strategy not in ["random", "constrained"]:
            raise ValueError("Strategy should be either random or constrained")
        self.optimization_order = optimization_order
        self.strategy = strategy
        self.max_steps = max_steps
        self.ckpt_path = ckpt_path
        self.adaltask = adaltask
        self.num_workers = num_workers
        self.train_loader = train_loader
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_val_score_threshold = batch_val_score_threshold
        self.correct_val_score_threshold = correct_val_score_threshold
        self.max_error_samples = max_error_samples
        self.max_correct_samples = max_correct_samples
        self.max_proposals_per_step = max_proposals_per_step

        self._subset_effect_count = {"pass": 0, "fail": 0}
        self._fullset_effect_count = {"pass": 0, "fail": 0}
        self._valset_effect_count = {"pass": 0, "fail": 0}
        self._demo_valset_effect_count = {"pass": 0, "fail": 0}
        self._effective_measure = {
            "subset": self._subset_effect_count,
            "fullset": self._fullset_effect_count,
            "valset": self._valset_effect_count,
            "demo_valset": self._demo_valset_effect_count,
        }
        self._raw_shots = raw_shots
        self._bootstrap_shots = bootstrap_shots
        self.demo_optimizers: List[DemoOptimizer] = []
        self.text_optimizers: List[TextOptimizer] = []
        self.save_traces = save_traces
        self.train_batch_size = train_batch_size
        self.weighted_sampling = weighted_sampling
        self.debug = debug
        self.exclude_input_fields_from_bootstrap_demos = (
            exclude_input_fields_from_bootstrap_demos
        )
        self.sequential_order = sequential_order
        self.skip_subset_val = skip_subset_val
        self.disable_backward_gradients = disable_backward_gradients
        self.disable_backward = disable_backward
        self.text_optimizers_config_kwargs = text_optimizers_config_kwargs

    def set_random_seed(self, seed: int):
        self.random_seed = seed

    # TODO: need to support checkpoint resume too!
    def diagnose(
        self, dataset: Any, split: str = "train", resume_from_ckpt: str = None
    ):
        """Run an evaluation on the trainset to track all error response, and its raw response using AdaplComponent's default configure_callbacks
        Args:
            dataset: Any: Dataset to evaluate
            split: str: Split name, default to train and it is also used as set the directory name for saving the logs
        Example:

        .. code-block:: python

            trainset, valset, testset = load_datasets(max_samples=10)
            adaltask = TGDWithEvalFnLoss(
                task_model_config=llama3_model,
                backward_engine_model_config=llama3_model,
                optimizer_model_config=llama3_model,
            )

            trainer = Trainer(adaltask=adaltask)
            diagnose = trainer.diagnose(dataset=trainset)
            print(diagnose)
        """
        # 1. track all intermediate outputs
        if resume_from_ckpt:
            self.resume_params_from_ckpt(resume_from_ckpt)
        self.adaltask.eval()
        if not self.ckpt_path:
            trainer_state = self.gather_trainer_states()
            self.prep_ckpt_file_path(trainer_state)
        printc(f"Checkpoint path: {self.ckpt_path}")
        save_path = os.path.join(self.ckpt_path, f"diagnose_{split}")
        logger.debug(f"Save diagnose to {save_path}")
        # One generator will be one file, all stats are in logger_metadata.json
        log_paths = self.adaltask.configure_callbacks(save_dir=save_path)
        # 2. evaluate
        acc = self.adaltask.validation_step(dataset, 0, self.num_workers)
        acc_score = acc.avg_score
        acc_per_item_scores = acc.per_item_scores
        # 3. load all completion from the log paths
        from adalflow.utils.file_io import load_jsonl, write_list_to_jsonl, save_json

        sorted_indices = sorted(
            range(len(acc_per_item_scores)), key=lambda i: acc_per_item_scores[i]
        )
        try:
            sorted_ids = [dataset[i].id for i in sorted_indices]
        except AttributeError:
            raise ValueError(
                "dataset should have an attribute id for tracking the samples"
            )
        logger.debug(f"sorted_indices: {sorted_indices}")

        sorted_scores = [acc_per_item_scores[i] for i in sorted_indices]
        logger.debug(f"sorted_scores: {sorted_scores}")
        sorted_dataset = [dataset[i] for i in sorted_indices]

        paths: Dict[str, List[str]] = {"Log": log_paths, "Diagnose": [], "Stats": []}

        # reorder the samples based on the score
        stats_list: List[Dict] = []
        for log_path in log_paths:
            stats_list = []
            file_name = os.path.basename(log_path)
            logger.debug(f"Loading log file: {file_name}")
            logs = load_jsonl(log_path)
            if not logs or len(logs) == 0:
                print(f"Log file {log_path} is empty. This llm is not called at all.")
                continue
            try:
                logs_dict = {log["output"]["id"]: log for log in logs}
            except KeyError:
                raise ValueError(
                    "Log file should have an output key with an id for tracking the samples. Ensure you have passed the data id to the Generator."
                )
            sorted_logs = [logs_dict[id] for id in sorted_ids]
            for log, score in zip(sorted_logs, sorted_scores):
                log["score"] = score
            write_list_to_jsonl(log_path, sorted_logs)

            log_dir = os.path.dirname(log_path)
            diagnose_filename = file_name.replace(".jsonl", "_diagnose.json")

            diagnose_file = os.path.join(log_dir, diagnose_filename)
            diagnose_items = []

            for i, log in enumerate(sorted_logs):
                if log["score"] < 0.5:
                    diagnose_item = {
                        "id": log["output"]["id"] if "id" in log["output"] else None,
                        "score": log["score"],
                        "prompt_kwargs": log["prompt_kwargs"],
                        "raw_response": log["output"]["raw_response"],
                        "answer": log["output"]["data"],
                        "dataset_item": sorted_dataset[i],
                        "error": log["output"]["error"],
                        "time_stamp": log["time_stamp"],
                    }
                    diagnose_items.append(diagnose_item)
            save_json(diagnose_items, diagnose_file)
            # save the stats
            stats = {
                "total_samples": len(sorted_logs),
                "total_error_samples": len(diagnose_items),
                "avg_score": acc_score,
            }
            stat_path = os.path.join(log_dir, "stats.json")
            save_json(stats, stat_path)
            logger.debug(f"Total error samples: {len(diagnose_items)}")
            logger.debug(f"Saved diagnose to {diagnose_file}")
            paths["Diagnose"].append(diagnose_file)
            paths["Stats"].append(stat_path)
            stats_list.append(stats)

        self.diagnose_report(
            split=split,
            acc_score=acc_score,
            stats_list=stats_list,
            log_paths=paths,
        )

    def diagnose_report(
        self,
        split: str,
        acc_score: Optional[float] = None,
        stats_list: Optional[List[Dict]] = None,
        log_paths: Optional[Dict[str, List[str]]] = None,
    ):
        import colorama
        from colorama import Fore

        # Initialize colorama
        colorama.init(autoreset=True)
        print(Fore.CYAN + "\n================== DIAGNOSE REPORT ==================\n")

        print(Fore.GREEN + f"✔ Split: {split}")

        # Check the accuracy score
        if acc_score is not None:
            print(Fore.GREEN + f"✔ Overall accuracy score: {acc_score:.2f}")
        else:
            print(Fore.RED + "✘ Accuracy score not provided or calculated.")

        # List the overall stats
        if stats_list is not None and len(stats_list) > 0:
            print(Fore.GREEN + "✔ Overall stats:")
            for idx, item in enumerate(stats_list):
                print(Fore.YELLOW + f"  - {idx + 1}: {item}")

        # Check for log paths
        if log_paths is not None:
            for key, paths in log_paths.items():
                if len(paths) > 0:
                    print(Fore.GREEN + f"✔ {key} paths:")
                    for idx, path in enumerate(paths):
                        print(Fore.YELLOW + f"  - {key} {idx + 1}: {path}")

        else:
            print(Fore.RED + "✘ No log paths available.")

        # General summary
        print(Fore.GREEN + "\n✔ Diagnose report completed successfully!")
        print(Fore.CYAN + "\n=====================================================\n")

    def debug_report(
        self,
        text_grad_debug_path: Optional[Dict[str, object]] = None,
        few_shot_demo_debug_path: Optional[Dict[str, object]] = None,
    ):
        import colorama
        from colorama import Fore

        # Initialize colorama
        colorama.init(autoreset=True)
        print(Fore.CYAN + "\n================== DEBUG REPORT ==================\n")

        if text_grad_debug_path:
            print(Fore.GREEN + f"✔ Text grad debug path: {text_grad_debug_path}")
        else:
            print(Fore.CYAN + "✘ Text grad debugging was not run.")

        if few_shot_demo_debug_path:
            print(
                Fore.GREEN + f"✔ Few shot demo debug path: {few_shot_demo_debug_path}"
            )
        else:
            print(Fore.RED + "✘ Few shot demo debugging was not run.")

        print(Fore.GREEN + "\n✔ The debug has run successfully!")
        print(
            Fore.YELLOW
            + "You can visualize the complete computation graph at the paths shown above."
        )
        print(Fore.CYAN + "\n===================================================\n")

    def resume_params_from_ckpt(self, ckpt_file: str):
        """Resume the parameters from the checkpoint file"""
        dict_data = load_standard_json(ckpt_file)
        # find the highest val score
        trainer_results: TrainerResult = TrainerResult.from_dict(dict_data)
        # restore the prompts to the adaltask
        val_scores = []
        # test_scores = []
        for step in trainer_results.step_results:
            if step.val_score:
                val_scores.append(step.val_score)
            # if step.test_score:
            #     test_scores.append(step.test_score)
        result_from_step = 0
        # if test_scores:
        #     result_from_step = test_scores.index(max(test_scores))
        if val_scores:
            printc(f"Val scores: {val_scores}")
            result_from_step = val_scores.index(max(val_scores))
        prompts: List[PromptData] = trainer_results.step_results[
            result_from_step
        ].prompt

        print(f"Restoring prompts: {prompts[0]}")

        self.adaltask._set_param_values(prompts)

    def fit(
        self,
        *,
        adaltask: Optional[AdalComponent] = None,
        train_loader: Optional[Any] = None,
        train_dataset: Optional[Any] = None,
        val_dataset: Optional[Any] = None,
        test_dataset: Optional[Any] = None,
        debug: bool = False,
        save_traces: bool = False,
        raw_shots: Optional[int] = None,
        bootstrap_shots: Optional[int] = None,
        resume_from_ckpt: Optional[
            str
        ] = None,  # TODO: have a more comprehensive ckpt loading in the future
        backward_pass_setup: Optional["BackwardPassSetup"] = None,
    ) -> Tuple[str, TrainerResult]:
        r"""
        train_loader: An iterable or collection of iterables specifying training samples.

        Returns:
            Tuple[str, TrainerResult]: Checkpoint file and the TrainerResult object
        """

        start_time = time.time()

        debug = debug or self.debug

        # check task
        adaltask = adaltask or self.adaltask
        self.adaltask = adaltask

        if not isinstance(adaltask, AdalComponent):
            raise ValueError(
                f"Task should be an instance of AdalComponent. Got {adaltask}"
            )
        raw_shots = raw_shots or self._raw_shots
        bootstrap_shots = bootstrap_shots or self._bootstrap_shots
        print(f"raw_shots: {raw_shots}, bootstrap_shots: {bootstrap_shots}")
        self.save_traces = save_traces or self.save_traces

        train_loader = train_loader or self.train_loader
        train_dataset = train_dataset or self.train_dataset

        if not train_loader and not train_dataset:
            raise ValueError("train_loader or train_dataset should be provided")

        if not train_loader and train_dataset:
            batch_size = self.train_batch_size

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True if not debug else False,
                seed=self.random_seed,
            )
        val_dataset = val_dataset or self.val_dataset

        test_dataset = test_dataset or self.test_dataset

        if not val_dataset:
            raise ValueError("val_dataset should be provided")
        # check train_loader and val_dataset and test_dataset, reject tuple
        if train_loader:
            exam_batch = next(iter(train_loader))
            if isinstance(exam_batch, tuple):
                raise ValueError(
                    "train_loader should return not be tuple, please use dict or a dataclass or with DataClass"
                )
        if val_dataset:
            if isinstance(val_dataset, tuple):
                raise ValueError(
                    "val_dataset should not be tuple, please use dict or a dataclass or with DataClass"
                )
        if test_dataset:
            if isinstance(test_dataset, tuple):
                raise ValueError(
                    "test_dataset should not be tuple, please use dict or a dataclass or with DataClass"
                )
        if train_dataset:
            if isinstance(train_dataset, tuple):
                raise ValueError(
                    "train_dataset should not be tuple, please use dict or a dataclass or with DataClass"
                )
        #  prepare optimizers
        self.optimizers: List[Optimizer] = self.adaltask.configure_optimizers(
            **self.text_optimizers_config_kwargs
        )
        self.text_optimizers = [
            opt for opt in self.optimizers if isinstance(opt, TextOptimizer)
        ]
        self.demo_optimizers = [
            opt for opt in self.optimizers if isinstance(opt, DemoOptimizer)
        ]

        # config optimizers
        if len(self._get_trainable_demo_params()) > 0:

            for opt in self.demo_optimizers:
                opt.config_shots(raw_shots=raw_shots, bootstrap_shots=bootstrap_shots)
                opt.use_weighted_sampling(weighted=self.weighted_sampling)
                opt.exclude_input_fields_from_bootstrap_demos = (
                    self.exclude_input_fields_from_bootstrap_demos
                )
            self.adaltask.configure_teacher_generator()
            print("Configured demo optimizers")
        else:
            print("No trainable demo params to optimize")
            self.demo_optimizers = []

        # configure backward engine
        if len(self._get_trainable_text_params()) > 0:

            if self.adaltask.backward_engine is None:
                self.adaltask.configure_backward_engine(
                    backward_pass_setup=backward_pass_setup
                )
                if self.disable_backward_gradients:
                    printc(
                        "Disabling backward engine for computing gradients, but can still run backward"
                    )
                    self.adaltask.disable_backward_engine()
        else:
            print("No trainable text params to optimize")
            self.text_optimizers = []

        if len(self.demo_optimizers) == 0 and len(self.text_optimizers) == 0:
            print("No trainable parameters to optimize")
            return None

        trainer_results = None
        starting_step = 0
        if resume_from_ckpt:
            self.ckpt_file = resume_from_ckpt
            self.ckpt_path = os.path.dirname(self.ckpt_file)
            dict_data = load_standard_json(self.ckpt_file)
            trainer_results: TrainerResult = TrainerResult.from_dict(dict_data)
            # restore the prompts to the adaltask
            val_scores = []
            for step in trainer_results.step_results:
                if step.val_score:
                    val_scores.append(step.val_score)
            result_from_step = 0
            if val_scores:
                printc(f"Val scores: {val_scores}")
                result_from_step = val_scores.index(max(val_scores))
            prompts: List[PromptData] = trainer_results.step_results[
                result_from_step
            ].prompt

            print(f"Restoring prompts: {prompts[0]}")

            self.adaltask._set_param_values(prompts)
            starting_step = len(trainer_results.steps) - 1
            self._add_history_text_optimizers(max(val_scores))

        else:
            trainer_results = (
                self._pre_fit(val_dataset, test_dataset)
                if trainer_results is None
                else trainer_results
            )

        if debug:
            print("Debugging mode")
            text_grad_debug_path, few_shot_demo_debug_path = None, None
            if (
                len(self.text_optimizers) > 0
                and len(self._get_trainable_text_params()) > 0
            ):
                text_grad_debug_path = self._fit_text_grads_one_step_for_debug(
                    train_loader
                )

            if (
                len(self.demo_optimizers) > 0
                and len(self._get_trainable_demo_params()) > 0
            ):
                few_shot_demo_debug_path = self._fit_demos_one_step_for_debug(
                    train_loader, train_dataset, val_dataset, test_dataset
                )
            self.debug_report(text_grad_debug_path, few_shot_demo_debug_path)
            return self.ckpt_file, trainer_results

        ########Run text_optimizers and demo optimizers in sequential order ########
        if (
            self.optimization_order == "mix"
            and len(self.demo_optimizers) > 0
            and len(self.text_optimizers) > 0
        ):
            if self.strategy == "random":
                self._fit_text_grad_demo_mix_random(
                    train_loader,
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    trainer_results,
                    starting_step=starting_step,
                )
            elif self.strategy == "constrained":
                self._fit_text_grad_demo_mix_constrained(
                    train_loader,
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    trainer_results,
                    starting_step=starting_step,
                )
            else:
                raise ValueError(f"Strategy {self.strategy} not supported")

        else:  # sequential, text first and demo second

            def run_text_optimizers(starting_step: int, trainer_results: TrainerResult):
                if len(self.text_optimizers) > 0:
                    if self.strategy == "random":
                        self._fit_text_grad_random(
                            train_loader,
                            val_dataset,
                            test_dataset,
                            trainer_results,
                            starting_step=starting_step,
                        )
                    elif self.strategy == "constrained":
                        # self.adaltask.configure_teacher_generator()  # use teacher as bootstrap intemediate results
                        self._fit_text_grad_constraint(
                            train_loader,
                            val_dataset,
                            test_dataset,
                            trainer_results=trainer_results,
                            starting_step=starting_step,
                        )
                    else:
                        raise ValueError(f"Strategy {self.strategy} not supported")

            def run_demo_optimizers(starting_step: int, trainer_results: TrainerResult):
                if len(self.demo_optimizers) > 0:
                    self.adaltask.configure_teacher_generator()
                    self.adaltask.disable_backward_engine()  # disable it to avoid backward engine for gradients
                    self._fit_demos_random(
                        train_loader,
                        train_dataset,
                        val_dataset,
                        test_dataset,
                        trainer_results=trainer_results,
                        starting_step=starting_step,
                    )

            if self.sequential_order == ["text", "demo"]:
                run_text_optimizers(starting_step, trainer_results)
                starting_step += self.max_steps
                print(f"Starting step: {starting_step}")
                print("steps", trainer_results.steps)
                run_demo_optimizers(starting_step, trainer_results)
            else:
                run_demo_optimizers(starting_step, trainer_results)
                starting_step += self.max_steps
                run_text_optimizers(starting_step, trainer_results)

        end_time = time.time()
        print(f"Training time: {end_time - start_time}s")
        trainer_results.total_time = end_time - start_time
        # test at the end
        if test_dataset:
            test_output = self.adaltask.validation_step(
                test_dataset, 0, self.num_workers
            )
            test_score = test_output.avg_score
            trainer_results.test_score = test_score
        # write the results to the checkpoint file
        save_json(trainer_results.to_dict(), self.ckpt_file)

        print(f"ckpt_file: {self.ckpt_file}")
        return self.ckpt_file, trainer_results

    @staticmethod
    def _estimate_num_epochs(train_loader: Any, max_steps: int):
        num_samples = len(train_loader)
        return max_steps // num_samples + 1

    def initial_validation(self, val_dataset: Any, test_dataset: Any):

        val_output = self.adaltask.validation_step(val_dataset, 0, self.num_workers)
        val_score = val_output.avg_score
        test_score = None
        if test_dataset is not None:
            test_output = self.adaltask.validation_step(
                test_dataset, 0, self.num_workers
            )
            test_score = test_output.avg_score
        trainer_results = TrainerResult(
            steps=[], val_scores=[], test_scores=[], step_results=[]
        )
        trainer_results.val_scores.append(val_score)
        trainer_results.test_scores.append(test_score)
        prompts = self.adaltask._get_param_values()
        # trainer_results.prompts.append(prompts)
        trainer_results.steps.append(0)
        # add step result
        step_result = TrainerStepResult(
            step=0,
            val_score=val_score,
            test_score=test_score,
            prompt=prompts,
        )
        trainer_results.step_results.append(step_result)
        print(f"Initial validation score: {val_score}")
        print(f"Initial test score: {test_score}")
        return trainer_results

    def gather_trainer_states(self):
        trainer_state = {}
        trainer_state["strategy"] = self.strategy
        trainer_state["demo_optimizers"] = self._get_trainable_demo_params()
        trainer_state["text_optimizers"] = self._get_trainable_text_params()
        trainer_state["max_steps"] = self.max_steps
        trainer_state["num_workers"] = self.num_workers
        trainer_state["raw_shots"] = self._raw_shots
        trainer_state["bootstrap_shots"] = self._bootstrap_shots
        trainer_state["weighted_sampling"] = self.weighted_sampling
        trainer_state["exclude_input_fields_from_bootstrap_demos"] = (
            self.exclude_input_fields_from_bootstrap_demos
        )
        trainer_state["batch_size"] = (
            self.train_loader.batch_size if self.train_loader else None
        )
        trainer_state["train_size"] = (
            len(self.train_loader.dataset) if self.train_loader else None
        )
        trainer_state["val_size"] = len(self.val_dataset) if self.val_dataset else None
        trainer_state["test_size"] = (
            len(self.test_dataset) if self.test_dataset else None
        )
        trainer_state["task_class"] = self.adaltask.__class__.__name__

        from adalflow.utils.serialization import serialize

        hash_key = hash_text_sha1(serialize(trainer_state))[0:5]
        trainer_state["hash_key"] = hash_key
        trainer_state["task_state_dict"] = self.adaltask.to_dict()
        # trainer_state["text_optimizers"] = [
        #     opt.to_dict() for opt in self.text_optimizers
        # ]
        # restore_state = AdalComponent.from_dict(
        #     trainer_state["task_state_dict"]
        # )  # tODO: add a test for adalcomponent
        # print(
        #     f"restore_state: {str(restore_state.to_dict()) == str(self.adaltask.to_dict())}"
        # )
        # print(f"task_state_dict: {trainer_state['task_state_dict']}")
        return trainer_state

    def prep_ckpt_file_path(self, trainer_state: Dict[str, Any] = None):
        r"""Prepare the checkpoint root path: ~/.adalflow/ckpt/task_name/.

        It also generates a unique checkpoint file name based on the strategy, max_steps, and a unique hash key.
        For multiple runs but with the same adalcomponent + trainer setup, the run number will be incremented.
        """
        if self.ckpt_file:
            return
        from adalflow.utils.global_config import get_adalflow_default_root_path

        if self.ckpt_path is None:
            default_root_path = get_adalflow_default_root_path()
            self.ckpt_path = os.path.join(
                default_root_path, "ckpt", self.adaltask.__class__.__name__
            )
            logger.debug(f"Checkpoint path: {self.ckpt_path}")
        os.makedirs(self.ckpt_path, exist_ok=True)
        # list all existing checkpoints with the same file name prefix
        hash_key = (
            trainer_state["hash_key"]
            if trainer_state and "hash_key" in trainer_state
            else str(uuid.uuid4())
        )
        file_name_prefix = f"{self.strategy}_max_steps_{self.max_steps}_{hash_key}"
        ckpt_files = [
            f for f in os.listdir(self.ckpt_path) if f.startswith(file_name_prefix)
        ]
        run: int = 1

        if ckpt_files:
            # Sort files based on last modification time
            ckpt_files.sort(
                key=lambda x: os.path.getmtime(os.path.join(self.ckpt_path, x)),
                reverse=True,
            )
            latest_ckpt_file = ckpt_files[0]
            # get the run number
            run = int(latest_ckpt_file.split("_run_")[-1].split(".json")[0]) + 1
        else:
            latest_ckpt_file = None

        self.ckpt_file = os.path.join(
            self.ckpt_path, f"{file_name_prefix}_run_{run}.json"
        )

    def _pre_fit(self, val_dataset: Any, test_dataset: Any) -> TrainerResult:
        # validate first (separate into another function where we can even save the outputs so that we can highlight error predictions)

        trainer_state = self.gather_trainer_states()
        trainer_results: TrainerResult = self.initial_validation(
            val_dataset, test_dataset
        )
        self._add_history_text_optimizers(trainer_results.val_scores[-1])
        trainer_results.trainer_state = trainer_state
        self.prep_ckpt_file_path(trainer_state)
        return trainer_results
        # end of validation

    def _fit_demos_one_step_for_debug(
        self, train_loader, train_dataset: Any, val_dataset: Any, test_dataset: Any
    ) -> Dict[str, object]:
        """Trace both the teacher and the student demos with scores and for sampling.
        For demos: we need to run both the teacher mode and the student mode."""

        # get_logger(level="DEBUG")
        print("Fitting using Random Demo Optimizer")
        self.prep_ckpt_file_path()
        debug_path = os.path.join(self.ckpt_path, "debug_demos")
        os.makedirs(debug_path, exist_ok=True)
        print(f"_fit_demos_one_step_for_debug save to {debug_path}")

        self.adaltask.train()
        self.adaltask.trace()
        self._set_demo_optimizers_dataset(train_dataset)

        # test teacher mode
        self.adaltask.use_teacher()
        train_loader.batch_size = 2
        pred_teacher = set()  # id of the teacher predictions
        batch = next(iter(train_loader))
        y_preds: List[Parameter] = self.adaltask.train_step(batch, 0, self.num_workers)
        if len(y_preds) != 2:
            raise ValueError("Expected 2 y_preds")
        nodes: List[Parameter] = y_preds[0].trace_graph(y_preds[0])[0]
        demo_params = [p for p in nodes if p.param_type == ParameterType.DEMOS]

        if len(demo_params) == 0:
            raise ValueError("No demo params found")

        if len(demo_params[0]._traces) != 2:
            raise ValueError(
                f"Expected 2 traces, got {len(demo_params[0]._traces)}, traces: {demo_params[0]._traces}"
            )

        # print(f"Teacher y_preds: {y_preds[0].to_dict()}")

        y_preds_outputs = [p.data for p in y_preds]

        batch_eval: EvaluationResult = self.adaltask.evaluate_samples(
            batch, y_preds_outputs
        )
        batch_acc = batch_eval.avg_score
        batch_per_item_scores = batch_eval.per_item_scores
        print(
            f"Validation accuracy: {batch_acc}, per item scores: {batch_per_item_scores}"
        )

        # test loss
        losses: List[Parameter] = self.adaltask.loss_step(
            batch, y_preds, 0, self.num_workers
        )
        # print(f"Losses: {losses[0].to_dict()}")
        self._demo_optimizers_add_scores(
            [sample.id for sample in batch], batch_per_item_scores, is_teacher=True
        )
        losses[0].backward()
        losses[1].backward()
        pred_teacher.add(batch[0].id)
        pred_teacher.add(batch[1].id)
        graph_path = os.path.join(debug_path, "graph")

        print(f"Graph saved to {graph_path}")

        # check the score of one param
        for key, val in demo_params[0]._traces.items():
            print(f"param: {key}, {demo_params[0].name}, val: {val}")
            score = val.score
            if score is None:
                raise ValueError("Score is None")
            print(f"param: {key}, score: {score}")
        # print(f"Loss after backward: {losses[0].to_dict()}")

        # tracking the bootstrap so we wont repeat the same samples

        # 2. run student mode

        demo_debug_result_path = None

        for batch_idx, batch in enumerate(train_loader):
            print(f"Training step: {batch_idx}")
            if batch_idx > 0:
                break
            # eval_student_mode
            self.adaltask.use_teacher(False)
            y_preds_student = self.adaltask.train_step(
                batch, batch_idx, self.num_workers
            )
            losses_student: List[Parameter] = self.adaltask.loss_step(  # noqa F841
                batch, y_preds_student, batch_idx, self.num_workers
            )

            self._demo_optimizers_add_scores(
                [sample.id for sample in batch], batch_per_item_scores, is_teacher=False
            )

            # Check the eval result
            y_preds_outputs = [p.data for p in y_preds_student]
            eval_result = self.adaltask.evaluate_samples(batch, y_preds_outputs)
            print(f"Eval result: {eval_result.avg_score}")

            # loss_students backward
            for loss in losses_student:
                loss.backward()

            # propose
            self._demo_optimizers_propose()
            graph_path = os.path.join(debug_path, "student_graph")

            demo_debug_result_path = losses_student[0].draw_graph(
                filepath=graph_path
            )  # noqa F841

            # test step
            self._demo_optimizers_step()

            for opt in self.demo_optimizers:
                if opt.proposing:
                    raise ValueError("Optimizer is still proposing")
            # check demo params
            opt_params = []
            for opt in self.demo_optimizers:
                opt_params.extend(opt.params)
            # print(f"Opt params: {opt_params}")
            for name, param in self.adaltask.named_parameters():

                if param.param_type == ParameterType.DEMOS:
                    print(
                        f"Demo param: {name}, value: {param.data}, param: {param.name}"
                    )
                    if param.data is None:
                        raise ValueError("Demo param data is None")

                    if len(param._traces) == 0:
                        raise ValueError(f"No traces found, param_id: {param.id}")
                    if len(param._previous_demos) > 0:
                        raise ValueError(
                            f"Previous demos should be empty, param: {param.id}"
                        )
                    if len(param._demos) == 0:
                        raise ValueError(f"No demos found, param: {param}")

        return demo_debug_result_path

    def _fit_text_grads_one_step_for_debug(self, train_loader: Any) -> Dict[str, str]:
        printc(
            "Debugging fitting one step with batch size 2 for text optimizer", "blue"
        )

        self.prep_ckpt_file_path()
        debug_path = os.path.join(self.ckpt_path, "debug_text_grads")
        os.makedirs(debug_path, exist_ok=True)
        train_loader.batch_size = 2
        train_loader.shuffle = True
        self.adaltask.train()  # this will turn everything to train mode
        correct_loss = None
        failed_loss = None
        all_losses = []
        printc("Finding one successful and one failed example", "blue")
        for batch in train_loader:
            y_preds = self.adaltask.train_step(batch, 0, self.num_workers)
            losses = self.adaltask.loss_step(batch, y_preds, 0, self.num_workers)
            # Collect all losses
            all_losses.extend(losses)
            for loss in losses:
                if loss.data > 0.5:
                    correct_loss = loss
                else:
                    failed_loss = loss
            if correct_loss is not None and failed_loss is not None:
                printc("Found correct and failed example", "blue")
                break
        if not all_losses:
            raise ValueError("No losses found in the dataset.")
        # Handle case where one or both losses are None
        if correct_loss is None or failed_loss is None:
            # Sort all_losses by their data values
            all_losses.sort(key=lambda x: x.data, reverse=True)  # Highest to lowest

            # Assign first and last loss in sorted list
            correct_loss = all_losses[0]
            failed_loss = all_losses[-1]
            print("Assigned correct_loss and failed_loss from sorted losses.")

        total_loss = sum_ops([copy(correct_loss), copy(failed_loss)])

        t0 = time.time()

        total_loss.backward()
        t1 = time.time()
        printc(f"finish loss backward in {t1-t0} seconds")
        # test optimizer
        self._propose_text_optimizers()
        t2 = time.time()
        printc(f"finish text optimizer step in {t2-t1} seconds")

        debug_files: Dict = total_loss.draw_graph(filepath=debug_path, full_trace=True)
        t3 = time.time()
        printc(f"finish draw_graph step in {t3-t2} seconds")
        debug_output_file = total_loss.draw_output_subgraph(filepath=debug_path)
        t4 = time.time()
        printc(f"finish draw_output_subgraph step in {t4-t3} seconds")
        debug_component_file = total_loss.draw_component_subgraph(filepath=debug_path)
        debug_files.update(debug_output_file)
        debug_files.update(debug_component_file)

        # zero grad
        self._zero_grad_text_optimizers()
        # revert
        self._revert_text_optimizers()

        total_loss.reset_all_gradients()

        # draw graph on a single loss
        total_loss = sum_ops([copy(failed_loss)])
        total_loss.backward()
        self._propose_text_optimizers()

        failed_debug_files = total_loss.draw_graph(
            filepath=debug_path, full_trace=False
        )
        failed_output_file = total_loss.draw_output_subgraph(filepath=debug_path)
        failed_component_file = total_loss.draw_component_subgraph(filepath=debug_path)
        failed_debug_files.update(failed_output_file)
        failed_debug_files.update(failed_component_file)

        for k, v in failed_debug_files.items():
            if k in debug_files:
                k = f"failed_{k}"
            debug_files[k] = v

        return debug_files

    def _set_demo_optimizers_dataset(self, train_dataset: Any):
        # init the dataset
        for opt in self.demo_optimizers:
            opt.set_dataset(train_dataset)

    def _demo_optimizers_propose(self):
        for opt in self.demo_optimizers:
            opt.propose()

    def _demo_optimizers_add_scores(
        self, ids: List[str], scores: List[float], is_teacher: bool = True
    ):
        for opt in self.demo_optimizers:
            # opt = cast(DemoOptimizer, opt)
            opt.add_scores(ids, scores, is_teacher)

    def _demo_optimizers_revert(self):
        for opt in self.demo_optimizers:
            opt.revert()

    def _demo_optimizers_step(self):
        for opt in self.demo_optimizers:
            opt.step()

    def _init_demo_optimizers(self):
        # init the dataset
        for opt in self.demo_optimizers:
            opt.init_shots()

    def _get_trainable_demo_params(self):
        params = []
        for opt in self.demo_optimizers:
            params.extend([p for p in opt.params if p.requires_opt])
        return params

    def _zero_grad_text_optimizers(self):
        for text_optimizer in self.text_optimizers:
            text_optimizer.zero_grad()

    def _text_optimizers_set_target_param(self):
        for text_optimizer in self.text_optimizers:
            text_optimizer.set_target_param()

    def _propose_text_optimizers(self):
        for text_optimizer in self.text_optimizers:
            text_optimizer.propose()

    def _add_failed_proposals_text_optimizers(self):
        for opt in self.text_optimizers:
            opt.add_failed_proposal()

    def _get_trainable_text_params(self):
        params = []
        for opt in self.text_optimizers:
            params.extend([p for p in opt.params if p.requires_opt])
        return params

    def _step_text_optimizers(self):
        for text_optimizer in self.text_optimizers:
            text_optimizer.step()

    def _add_history_text_optimizers(self, val_score: float):
        if not isinstance(val_score, float):
            raise ValueError(
                f"val_score should be a float, got {type(val_score)}, {val_score}"
            )
        for text_optimizer in self.text_optimizers:
            text_optimizer.add_score_to_params(round(val_score, 4))

    def _revert_text_optimizers(self):
        for text_optimizer in self.text_optimizers:
            text_optimizer.revert()

    def _increment_step_from_last_improvement_text_optimizers(self):
        for text_optimizer in self.text_optimizers:
            text_optimizer.increment_steps_from_last_improvement()

    def _reset_steps_from_last_improvement_text_optimizers(self):
        for text_optimizer in self.text_optimizers:
            text_optimizer.reset_steps_from_last_improvement()

    def _check_optimizer_proposal(self):
        r"""Return True if all optimizers have proposed a new prompt"""
        for text_optimizer in self.text_optimizers:
            if not text_optimizer.proposing:
                return False
        return True

    # TODO: mix training teacher should keep updated with the new prompt
    def _fit_text_grad_demo_mix_constrained(
        self,
        train_loader: Any,
        train_dataset: Any,
        val_dataset: Any,
        test_dataset: Any,
        trainer_results: TrainerResult = None,
        starting_step: int = 0,
    ):
        from adalflow.optim.parameter import Parameter

        logger.info("Fitting using Textual Gradient Descent")
        trainer_results = (
            self._pre_fit(val_dataset, test_dataset)
            if trainer_results is None
            else trainer_results
        )
        print(f"_fit_text_grad_demo_mix_constrained save to {self.ckpt_file}")

        if train_dataset is None:
            raise ValueError("train_dataset is required")

        self.adaltask.train()
        self._zero_grad_text_optimizers()
        self._set_demo_optimizers_dataset(train_dataset)

        num_epochs = self._estimate_num_epochs(train_loader, self.max_steps)
        total_steps = starting_step
        teacher_losses_cache: Dict[str, Parameter] = {}
        all_samples, all_losses, all_y_preds = [], [], []
        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            for steps, batch in enumerate((pbar := tqdm(train_loader, position=0))):
                total_steps += 1
                if total_steps > self.max_steps + starting_step:
                    print("Reached max steps")
                    break
                self._zero_grad_text_optimizers()
                pbar.set_description(f"Training Step: {total_steps}")
                self.adaltask.train()  # this will turn everything to train mode
                self.adaltask.trace()  # NOTE: this needs to be turned on?
                self.adaltask.use_teacher(False)
                y_preds = self.adaltask.train_step(batch, steps, self.num_workers)
                losses = self.adaltask.loss_step(
                    batch, y_preds, steps, self.num_workers
                )
                # moving batch
                all_samples.extend(batch)
                all_losses.extend(losses)  # student losses
                # extract the non-parameter y_preds
                all_y_preds.extend(
                    [y.data for y in y_preds if isinstance(y, Parameter)]
                )

                # for loss in losses:
                #     loss.backward_engine_disabled = (
                #         True  # temporary disable the backward engine
                #     )
                #     loss.backward()
                # handle the demo
                print(f"batch: {batch}")
                self._demo_optimizers_add_scores(
                    [sample.id for sample in batch],
                    [float(loss.data) for loss in losses],
                    is_teacher=False,
                )
                # Trace the teacher run
                self.adaltask.use_teacher(True)
                self.adaltask.train()
                self.adaltask.trace()
                # filter by id
                batch_for_teacher = []
                for sample in batch:
                    if sample.id not in teacher_losses_cache:
                        batch_for_teacher.append(sample)

                y_preds_teacher = self.adaltask.train_step(
                    batch_for_teacher, total_steps, self.num_workers
                )
                losses_teacher: List[Parameter] = self.adaltask.loss_step(
                    batch_for_teacher, y_preds_teacher, total_steps, self.num_workers
                )
                self._demo_optimizers_add_scores(
                    [sample.id for sample in batch_for_teacher],
                    [float(loss.data) for loss in losses_teacher],
                    is_teacher=True,
                )
                for idx, (sample, loss) in enumerate(
                    zip(batch_for_teacher, losses_teacher)
                ):
                    teacher_losses_cache[sample.id] = loss

                all_samples, all_losses, all_y_preds = (
                    self._text_grad_constraint_propose_step(
                        steps=steps,
                        all_samples=all_samples,
                        all_losses=all_losses,
                        all_y_preds=all_y_preds,
                        include_demo_optimizers=True,
                        trainer_results=trainer_results,
                        val_dataset=val_dataset,
                        test_dataset=test_dataset,
                        total_steps=total_steps,
                    )
                )

                # if not self._check_optimizer_proposal():
                #     print(
                #         "No proposal can improve the subset and full set, go to next step"
                #     )
                #     # self._add_failed_proposals_text_optimizers()

                #     self._add_one_step_in_trainer_results(
                #         trainer_results,
                #         trainer_results.val_scores[-1],
                #         trainer_results.test_scores[-1],
                #         trainer_results.prompts[-1],
                #         total_steps,
                #     )

                #     continue

                # # set the batch size to the size of the validation set
                # last_val_score = trainer_results.val_scores[-1]
                # val_output = self.adaltask.validation_step(
                #     val_dataset,
                #     total_steps,
                #     self.num_workers,
                #     minimum_score=last_val_score,
                # )
                # val_score = val_output.avg_score
                # self._add_history_text_optimizers(val_score)

                # if val_score > last_val_score:
                #     print(f"Optimizer step: {val_score} > {last_val_score}")
                #     # self.optimizer.step()
                #     self._step_text_optimizers()
                #     self._demo_optimizers_step()

                #     # test the model
                #     test_score = None
                #     if test_dataset is not None:
                #         test_output = self.adaltask.validation_step(
                #             test_dataset, total_steps, self.num_workers
                #         )
                #         test_score = test_output.avg_score

                #     new_prompts = self.adaltask._get_param_values()
                #     self._add_one_step_in_trainer_results(
                #         trainer_results,
                #         val_score,
                #         test_score,
                #         new_prompts,
                #         total_steps,
                #     )
                #     all_samples, all_losses, all_y_preds = [], [], []
                # else:
                #     print(f"Optimizer revert: {val_score} <= {last_val_score}")
                #     # self.optimizer.revert()
                #     self._revert_text_optimizers()
                #     self._demo_optimizers_revert()
                #     # save the score, no change
                #     self._add_one_step_in_trainer_results(
                #         trainer_results,
                #         last_val_score,
                #         trainer_results.test_scores[-1],
                #         trainer_results.prompts[-1],
                #         total_steps,
                #         attempted_val_score=val_score,
                #     )

                # print(f"Saving checkpoint to {self.ckpt_file}")
                # save_json(trainer_results.to_dict(), self.ckpt_file)
            save_json(trainer_results.to_dict(), self.ckpt_file)  # checkpoint

    def _fit_text_grad_demo_mix_random(
        self,
        train_loader: Any,
        train_dataset: Any,
        val_dataset: Any,
        test_dataset: Any,
        train_results: TrainerResult = None,
        starting_step: int = 0,
    ):
        logger.info("Fitting using Textual Gradient Descent")

        trainer_results = (
            self._pre_fit(val_dataset, test_dataset)
            if train_results is None
            else train_results
        )
        print(f"_fit_text_grad_demo_mix_random save to {self.ckpt_file}")

        if train_dataset is None:
            raise ValueError("train_dataset is required")

        self.adaltask.train()
        self._zero_grad_text_optimizers()
        self._set_demo_optimizers_dataset(train_dataset)

        num_epochs = self._estimate_num_epochs(train_loader, self.max_steps)
        total_steps = starting_step
        teacher_losses_cache: Dict[str, Parameter] = {}
        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            for steps, batch in enumerate((pbar := tqdm(train_loader, position=0))):
                total_steps += 1
                if total_steps > self.max_steps + starting_step:
                    print("Reached max steps")
                    break
                self._zero_grad_text_optimizers()
                pbar.set_description(f"Training Step: {total_steps}")
                self.adaltask.train()  # this will turn everything to train mode
                self.adaltask.trace()  # NOTE: this needs to be turned on?
                self.adaltask.use_teacher(False)
                y_preds = self.adaltask.train_step(batch, steps, self.num_workers)
                losses = self.adaltask.loss_step(
                    batch, y_preds, steps, self.num_workers
                )
                total_loss = sum_ops(losses)
                print("Loss backward...")
                total_loss.backward()
                # for loss in losses:
                #     loss.backward_engine_disabled = (
                #         True  # temporary disable the backward engine
                #     )
                #     loss.backward()
                # handle the demo
                self._demo_optimizers_add_scores(
                    [sample.id for sample in batch],
                    [float(loss.data) for loss in losses],
                    is_teacher=False,
                )
                # Trace the teacher run
                self.adaltask.use_teacher(True)
                self.adaltask.train()
                self.adaltask.trace()
                # filter by id
                batch_for_teacher = []
                for sample in batch:
                    if sample.id not in teacher_losses_cache:
                        batch_for_teacher.append(sample)

                y_preds_teacher = self.adaltask.train_step(
                    batch_for_teacher, total_steps, self.num_workers
                )
                losses_teacher: List[Parameter] = self.adaltask.loss_step(
                    batch_for_teacher, y_preds_teacher, total_steps, self.num_workers
                )
                self._demo_optimizers_add_scores(
                    [sample.id for sample in batch_for_teacher],
                    [float(loss.data) for loss in losses_teacher],
                    is_teacher=True,
                )
                # for loss in losses_teacher:
                #     loss.backward_engine_disabled = (
                #         True  # temporary disable the backward engine
                #     )
                #     loss.backward()
                # save the teacher predictions, if Generator is in cache mode, it will also avoid re-running the teacher
                for idx, (sample, loss) in enumerate(
                    zip(batch_for_teacher, losses_teacher)
                ):
                    teacher_losses_cache[sample.id] = loss

                print("Optimizer propose...")
                self._propose_text_optimizers()
                self._demo_optimizers_propose()
                new_prompts = self.adaltask._get_param_values()
                print("New prompts: ", new_prompts)
                # set the batch size to the size of the validation set
                last_val_score = trainer_results.val_scores[-1]
                val_output = self.adaltask.validation_step(
                    val_dataset,
                    total_steps,
                    self.num_workers,
                    minimum_score=last_val_score,
                )
                val_score = val_output.avg_score
                self._add_history_text_optimizers(val_score)

                if val_score > last_val_score:
                    print(f"Optimizer step: {val_score} > {last_val_score}")
                    # self.optimizer.step()
                    self._step_text_optimizers()
                    self._demo_optimizers_step()

                    # test the model
                    test_score = None
                    # if test_dataset is not None:
                    #     test_output = self.adaltask.validation_step(
                    #         test_dataset, total_steps, self.num_workers
                    #     )
                    #     test_score = test_output.avg_score
                    self._add_one_step_in_trainer_results(
                        trainer_results,
                        val_score,
                        test_score,
                        new_prompts,
                        total_steps,
                    )
                else:
                    print(f"Optimizer revert: {val_score} <= {last_val_score}")
                    # self.optimizer.revert()
                    self._revert_text_optimizers()
                    self._demo_optimizers_revert()
                    # save the score, no change
                    self._add_one_step_in_trainer_results(
                        trainer_results,
                        last_val_score,
                        trainer_results.test_scores[-1],
                        trainer_results.step_results[-1].prompt,
                        total_steps,
                        attempted_val_score=val_score,
                    )

                print(f"Saving checkpoint to {self.ckpt_file}")
                save_json(trainer_results.to_dict(), self.ckpt_file)
            save_json(trainer_results.to_dict(), self.ckpt_file)  # checkpoint

    def _fit_demos_random(
        self,
        train_loader,
        train_dataset: Any,
        val_dataset: Any,
        test_dataset: Any,
        trainer_results: TrainerResult,
        starting_step: int,
    ):
        logger.info("Fitting using Random Demo Optimizer")
        # self.adaltask.train()
        trainer_results = (
            self._pre_fit(val_dataset, test_dataset)
            if trainer_results is None
            else trainer_results
        )
        print(f"_fit_demos_random save to {self.ckpt_file}")
        print(f"Starting step: {starting_step}")

        self.adaltask.train()
        self.adaltask.trace()
        self._set_demo_optimizers_dataset(train_dataset)

        # total_steps = 0
        train_loader.set_max_steps(self.max_steps)
        teacher_losses_cache: Dict[str, Parameter] = {}
        pbar = tqdm(
            zip(range(self.max_steps), train_loader), total=self.max_steps, desc="Step"
        )

        self.adaltask.disable_backward_engine()  # disable it to avoid backward engine for gradients

        for step, batch in pbar:
            step = step + starting_step + 1
            print(f"Training Step: {step}")
            pbar.set_description(f"Training Step: {step}")
            # Trace the run in the demos
            self.adaltask.train()
            self.adaltask.trace()
            self.adaltask.use_teacher(False)
            y_preds = self.adaltask.train_step(batch, step, self.num_workers)
            losses: List[Parameter] = self.adaltask.loss_step(
                batch, y_preds, step, self.num_workers
            )
            self._demo_optimizers_add_scores(
                [sample.id for sample in batch],
                [float(loss.data) for loss in losses],
                is_teacher=False,
            )

            for loss in losses:
                # loss.backward_engine_disabled = (
                #     True  # temporary disable the backward engine
                # )
                loss.backward()  # TODO: ensure no gradients in the backward, disable backward engine, trace the score to each class instead
            # Trace the teacher run
            self.adaltask.use_teacher(True)
            self.adaltask.train()
            self.adaltask.trace()
            # filter by id
            batch_for_teacher = []
            for sample in batch:
                if sample.id not in teacher_losses_cache:
                    batch_for_teacher.append(sample)

            y_preds_teacher = self.adaltask.train_step(
                batch_for_teacher, step, self.num_workers
            )
            losses_teacher: List[Parameter] = self.adaltask.loss_step(
                batch_for_teacher, y_preds_teacher, step, self.num_workers
            )
            self._demo_optimizers_add_scores(
                [sample.id for sample in batch_for_teacher],
                [float(loss.data) for loss in losses_teacher],
                is_teacher=True,
            )
            for loss in losses_teacher:
                loss.backward_engine_disabled = (
                    True  # temporary disable the backward engine
                )
                loss.backward()
            # save the teacher predictions, if Generator is in cache mode, it will also avoid re-running the teacher
            for idx, (sample, loss) in enumerate(
                zip(batch_for_teacher, losses_teacher)
            ):
                teacher_losses_cache[sample.id] = loss
            # propose
            self._demo_optimizers_propose()

            new_prompts = self.adaltask._get_param_values()
            print(f"New prompts: {new_prompts}")

            # validate
            if self.adaltask.validate_condition(step, total_steps=self.max_steps):
                last_val_score = trainer_results.val_scores[-1]
                val_output = self.adaltask.validation_step(
                    val_dataset,
                    step,
                    self.num_workers,
                    minimum_score=last_val_score,
                )
                val_score = val_output.avg_score

                if val_score > last_val_score:
                    print(
                        f"Pass validation: {val_score} > {trainer_results.val_scores[-1]}"
                    )
                    self._track_effectiveness("demo_valset", True)

                    self._demo_optimizers_step()
                    for opt in self.demo_optimizers:
                        if opt.proposing:
                            raise ValueError("Optimizer is still proposing")

                    # test the new prompts
                    test_score = None
                    # if test_dataset is not None:
                    #     test_output = self.adaltask.validation_step(
                    #         test_dataset, step, self.num_workers
                    #     )
                    #     test_score = test_output.avg_score
                    self._add_one_step_in_trainer_results(
                        trainer_results,
                        val_score,
                        test_score=test_score,
                        prompts=new_prompts,
                        step=step,
                        attempted_val_score=val_score,
                    )
                else:
                    self._track_effectiveness("demo_valset", False)
                    print(f"Fail validation: {val_score} <= {last_val_score}, revert")
                    self._demo_optimizers_revert()
                    # ensure all demo optimizer are not proposing
                    for opt in self.demo_optimizers:
                        if opt.proposing:
                            raise ValueError("Optimizer is still proposing")
                    self._add_one_step_in_trainer_results(
                        trainer_results,
                        last_val_score,
                        test_score=trainer_results.test_scores[-1],
                        prompts=trainer_results.step_results[-1].prompt,
                        step=step,
                        attempted_val_score=val_score,
                    )
                save_json(trainer_results.to_dict(), self.ckpt_file)
                pbar.update(1)
        self._compute_validate_stats(trainer_results)
        save_json(trainer_results.to_dict(), self.ckpt_file)
        if self.save_traces:
            for i, demo_opt in enumerate(self.demo_optimizers):
                for param in demo_opt.params:

                    teacher_traces = param._traces
                    student_traces = param._student_traces

                    trace_file = os.path.join(
                        self.ckpt_path,
                        f"opt_{i}_param_{param.name}_teacher_traces.json",
                    )
                    save_json(teacher_traces, trace_file)
                    trace_file = os.path.join(
                        self.ckpt_path,
                        f"opt_{i}_param_{param.name}_student_traces.json",
                    )
                    save_json(student_traces, trace_file)
                    # save demos
                    demo_file = os.path.join(
                        self.ckpt_path, f"opt_{i}_param_{param.name}_demos.json"
                    )
                    save_json(param._demos, demo_file)

        print(f"Saved ckpt to {self.ckpt_file}")
        return trainer_results

    @staticmethod
    def _compute_validate_stats(trainer_results: TrainerResult):
        attempted_val_scores = [
            (
                step_result.attempted_val_score
                if step_result.attempted_val_score is not None
                else step_result.val_score
            )
            for step_result in trainer_results.step_results
        ]
        array = np.array(attempted_val_scores)
        mean = round(float(np.mean(array)), 4)
        std = round(float(np.std(array)), 4)
        max_score = round(float(np.max(array)), 4)
        min_score = round(float(np.min(array)), 4)
        trainer_results.validate_stats = TrainerValidateStats(
            max_score=max_score,
            min_score=min_score,
            mean_of_score=mean,
            std_of_score=std,
        )

    def _random_propose_step(
        self,
        current_step: int,
        all_samples,
        all_losses: List["Parameter"],
        all_y_preds,
        trainer_results: TrainerResult = None,
        val_dataset: Any = None,
    ):
        """Handles a single training step in random batch"""

        tdqm_loader = tqdm(range(self.max_proposals_per_step), desc="Proposing")

        use_eval_loss_fn = False
        if self.adaltask.loss_eval_fn is not None:
            use_eval_loss_fn = True

        batch_score_list = self.adaltask.evaluate_samples(
            samples=all_samples, y_preds=all_y_preds, use_loss_eval_fn=use_eval_loss_fn
        )
        # scores that we will compare with
        batch_score = batch_score_list.avg_score
        last_val_score = trainer_results.val_scores[-1]
        val_score_increased = False
        val_score = None

        for i in tdqm_loader:
            print(f"Proposal: {i+1}")
            start_time = time.time()
            self._propose_text_optimizers()
            printc(f"Propose time: {time.time()-start_time}")
            new_prompts = self.adaltask._get_param_values()
            print("New prompts: ", new_prompts)

            # validate on the batch
            batch_val_score_list = self.adaltask.validation_step(
                all_samples,
                current_step,
                use_loss_eval_fn=use_eval_loss_fn,
            )
            batch_val_score = batch_val_score_list.avg_score

            if (
                batch_val_score == batch_score
                and batch_score >= self.batch_val_score_threshold
            ) or batch_val_score > batch_score:  # allow perfect subset to pass

                printc(
                    f"Pass subset check:{use_eval_loss_fn}, {batch_val_score} > {batch_score}"
                )
                self._track_effectiveness("subset", True)

            else:
                printc(
                    f"Fail subset check, try next proposal: {use_eval_loss_fn}, {batch_val_score} <= {batch_score}"
                )
                self._add_failed_proposals_text_optimizers()
                self._track_effectiveness("subset", False)
                self._revert_text_optimizers()
                continue

            # validate on the whole validation set
            # set the batch size to the size of the validation set
            val_output = self.adaltask.validation_step(
                val_dataset,
                current_step,
                self.num_workers,
                minimum_score=last_val_score,
            )
            val_score = val_output.avg_score

            if val_score > last_val_score:

                print(f"Optimizer step: {val_score} > {last_val_score}")
                # track the effectiveness
                self._track_effectiveness("valset", True)
                self._step_text_optimizers()
                self._add_history_text_optimizers(val_score)  # track top performor
                # test the model
                # test_output = self.adaltask.validation_step(
                #     test_dataset, total_steps, self.num_workers
                # )
                # test_score = test_output.avg_score
                test_score = None
                self._add_one_step_in_trainer_results(
                    trainer_results,
                    val_score,
                    test_score,
                    new_prompts,
                    current_step,
                )
                val_score_increased = True
                self._reset_steps_from_last_improvement_text_optimizers()
                break

            else:
                # if val_score < last_val_score:
                self._add_failed_proposals_text_optimizers()  # track failed proposals

                print(f"Optimizer revert: {val_score} <= {last_val_score}")
                self._revert_text_optimizers()
                self._track_effectiveness("valset", False)
                self._add_failed_proposals_text_optimizers()

                continue

        if not val_score_increased:
            print("No proposal can improve the subset and full set, and val set")
            self._zero_grad_text_optimizers()
            # save the score, no change
            self._add_one_step_in_trainer_results(
                trainer_results,
                last_val_score,
                trainer_results.test_scores[-1],
                trainer_results.step_results[-1].prompt,
                current_step,
                attempted_val_score=val_score,
            )
            self._increment_step_from_last_improvement_text_optimizers()
        print(f" {current_step}, Saving checkpoint to {self.ckpt_file}")
        trainer_results.effective_measure = self._effective_measure
        save_json(trainer_results.to_dict(), self.ckpt_file)

    def _fit_text_grad_random(
        self,
        train_loader: Any,
        val_dataset: Any,
        test_dataset: Any,
        trainer_results: TrainerResult = None,
        starting_step: int = 0,
    ) -> TrainerResult:
        logger.info("Fitting using Textual Gradient Descent")
        trainer_results = (
            self._pre_fit(val_dataset, test_dataset)
            if trainer_results is None
            else trainer_results
        )
        print(f"_fit_text_grad_random save to {self.ckpt_file}")

        self.adaltask.train()
        # self.optimizer.zero_grad()
        self._zero_grad_text_optimizers()

        num_epochs = self._estimate_num_epochs(train_loader, self.max_steps)
        print(f"num_epochs: {num_epochs}, max_steps: {self.max_steps}")
        current_step = starting_step
        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            print(f"Epoch: {epoch}")
            for steps, batch in enumerate((pbar := tqdm(train_loader, position=0))):
                current_step += 1
                if current_step > self.max_steps + starting_step:
                    print("Reached max steps")
                    break
                self._zero_grad_text_optimizers()
                pbar.set_description(f"Training Step: {current_step}")
                self.adaltask.train()  # this will turn everything to train mode
                try:

                    y_preds = self.adaltask.train_step(batch, steps, self.num_workers)
                except Exception as e:
                    print(f"Error in train step: {e}")
                    raise e
                try:
                    losses = self.adaltask.loss_step(
                        batch, y_preds, steps, self.num_workers
                    )
                except Exception as e:
                    print(f"Error in loss step: {e}")
                    raise e
                total_loss = sum_ops(losses)
                try:
                    if not self.disable_backward:
                        total_loss.backward()
                except Exception as e:
                    print(f"Error in backward: {e}")
                    raise e
                print("Optimizer propose...")

                all_y_preds = [
                    y.data for y in y_preds if isinstance(y, OutputParameter)
                ]
                self._random_propose_step(
                    current_step=current_step,
                    all_samples=batch,
                    all_losses=losses,
                    all_y_preds=all_y_preds,
                    trainer_results=trainer_results,
                    val_dataset=val_dataset,
                )
                # self._propose_text_optimizers()
                # new_prompts = self.adaltask._get_param_values()
                # print("New prompts: ", new_prompts)
                # # set the batch size to the size of the validation set
                # last_val_score = trainer_results.val_scores[-1]
                # val_output = self.adaltask.validation_step(
                #     val_dataset,
                #     current_step,
                #     self.num_workers,
                #     minimum_score=last_val_score,
                # )
                # val_score = val_output.avg_score

                # if val_score > last_val_score:

                #     print(f"Optimizer step: {val_score} > {last_val_score}")
                #     # track the effectiveness
                #     self._track_effectiveness("valset", True)
                #     # self.optimizer.step()
                #     self._step_text_optimizers()
                #     self._add_history_text_optimizers(val_score)  # track top performor
                #     # test the model
                #     # test_output = self.adaltask.validation_step(
                #     #     test_dataset, total_steps, self.num_workers
                #     # )
                #     # test_score = test_output.avg_score
                #     test_score = None
                #     self._add_one_step_in_trainer_results(
                #         trainer_results,
                #         val_score,
                #         test_score,
                #         new_prompts,
                #         current_step,
                #     )
                # else:
                #     # if val_score < last_val_score:
                #     self._add_failed_proposals_text_optimizers()  # track failed proposals

                #     print(f"Optimizer revert: {val_score} <= {last_val_score}")
                #     self._revert_text_optimizers()
                #     self._track_effectiveness("valset", False)
                #     # save the score, no change
                #     self._add_one_step_in_trainer_results(
                #         trainer_results,
                #         last_val_score,
                #         trainer_results.test_scores[-1],
                #         trainer_results.prompts[-1],
                #         current_step,
                #         attempted_val_score=val_score,
                #     )

                # print(f" {current_step}, Saving checkpoint to {self.ckpt_file}")
                # save_json(trainer_results.to_dict(), self.ckpt_file)
            save_json(trainer_results.to_dict(), self.ckpt_file)  # checkpoint
        return trainer_results

    @staticmethod
    def _add_one_step_in_trainer_results(
        trainer_results: TrainerResult,
        val_score: float,
        test_score: float,
        prompts: List[PromptData],  # target prompts
        step: int,
        attempted_val_score: Optional[float] = None,
    ):

        step_results = TrainerStepResult(
            step=step,
            val_score=val_score,
            test_score=test_score,
            prompt=prompts,
            attempted_val_score=attempted_val_score,
        )
        trainer_results.step_results.append(step_results)

        trainer_results.val_scores.append(val_score)
        trainer_results.test_scores.append(test_score)
        trainer_results.steps.append(step)

    # def _downsample_move_batch(
    #     self, all_samples, all_losses: List["Parameter"], all_y_preds, acc_score_list
    # ):
    #     """Downsample the moving batch to a more balanced error and correct samples"""

    #     from adalflow.optim.parameter import Parameter

    #     if not all([score >= 0 and score <= 1 for score in acc_score_list]):
    #         raise ValueError(
    #             "acc_score_list should only contain values between 0 and 1"
    #         )

    #     for loss in all_losses:
    #         if not isinstance(loss, Parameter):
    #             raise ValueError("Loss should be a Parameter object")
    #     max_moving_batch_size = 20

    #     correct_indices = [i for i, score in enumerate(acc_score_list) if score > 0.5]
    #     error_indices = [i for i, score in enumerate(acc_score_list) if score <= 0.5]

    #     if (
    #         len(error_indices) + len(correct_indices)
    #         <= max_moving_batch_size
    #         # and len(correct_indices) <= max_moving_batch_size
    #     ):
    #         return all_samples, all_losses, all_y_preds, acc_score_list

    #     # downsample from all samples
    #     new_sample_indices = random.sample(
    #         range(len(all_samples)), min(max_moving_batch_size, len(all_samples))
    #     )
    #     all_samples = [all_samples[i] for i in new_sample_indices]
    #     all_losses = [all_losses[i] for i in new_sample_indices]
    #     all_y_preds = [all_y_preds[i] for i in new_sample_indices]
    #     acc_score_list = [acc_score_list[i] for i in new_sample_indices]
    #     return all_samples, all_losses, all_y_preds, acc_score_list

    def _downsample_move_batch(
        self, all_samples, all_losses: List["Parameter"], all_y_preds, acc_score_list
    ):
        """Downsample the moving batch to a more balanced error and correct samples"""

        from adalflow.optim.parameter import Parameter

        if not all([score >= 0 and score <= 1 for score in acc_score_list]):
            raise ValueError(
                "acc_score_list should only contain values between 0 and 1"
            )

        for loss in all_losses:
            if not isinstance(loss, Parameter):
                raise ValueError("Loss should be a Parameter object")

        max_moving_batch_size = 20
        min_error_samples = 4

        correct_indices = [i for i, score in enumerate(acc_score_list) if score > 0.5]
        error_indices = [i for i, score in enumerate(acc_score_list) if score <= 0.5]

        if (
            len(error_indices) + len(correct_indices)
            <= max_moving_batch_size
            # and len(correct_indices) <= max_moving_batch_size
        ):
            return all_samples, all_losses, all_y_preds, acc_score_list

        # Adjust downsampling logic
        if len(error_indices) < min_error_samples:
            remaining_capacity = max_moving_batch_size - len(error_indices)
            correct_indices = random.sample(correct_indices, max(0, remaining_capacity))
        else:
            # Set aside minimum error samples
            retained_error_indices = error_indices[:min_error_samples]
            remaining_error_indices = error_indices[min_error_samples:]

            # Combine remaining error and correct indices for unified sampling
            combined_indices = remaining_error_indices + correct_indices
            sampled_combined_indices = random.sample(
                combined_indices, max(0, max_moving_batch_size - min_error_samples)
            )

            error_indices = retained_error_indices
            correct_indices = [
                i for i in sampled_combined_indices if i in correct_indices
            ]
            remaining_error_indices = [
                i for i in sampled_combined_indices if i in remaining_error_indices
            ]
            error_indices += remaining_error_indices

        error_samples = [all_samples[i] for i in error_indices]
        error_losses = [all_losses[i] for i in error_indices]
        error_y_preds = [all_y_preds[i] for i in error_indices]
        error_scores = [acc_score_list[i] for i in error_indices]

        correct_samples = [all_samples[i] for i in correct_indices]
        correct_losses = [all_losses[i] for i in correct_indices]
        correct_y_preds = [all_y_preds[i] for i in correct_indices]
        correct_scores = [acc_score_list[i] for i in correct_indices]

        # Combine error and downsampled correct samples
        all_samples = error_samples + correct_samples
        all_losses = error_losses + correct_losses
        all_y_preds = error_y_preds + correct_y_preds
        acc_score_list = error_scores + correct_scores

        return all_samples, all_losses, all_y_preds, acc_score_list

    def _moving_batch_sample(
        self, acc_score_list: List[float]
    ) -> Tuple[float, List[int]]:
        """Sample from both correct and error samples according to max_error_samples and max_correct_samples"""
        # ensure only 0 and 1 in the acc_score_list
        import numpy as np

        if not all(0 <= score <= 1 for score in acc_score_list):
            raise ValueError("acc_score_list should only contain 0 and 1")
        correct_indices = [
            i
            for i, score in enumerate(acc_score_list)
            if score > self.correct_val_score_threshold
        ]
        error_indices = [
            i
            for i, score in enumerate(acc_score_list)
            if score <= self.correct_val_score_threshold
        ]
        print(f"Moving batch correct size: {len(correct_indices)}")
        print(f"Moving batch error size: {len(error_indices)}")
        # if len(error_indices) == 0:
        #     raise ValueError("No error samples found")
        sampled_error_indices = random.sample(
            error_indices, min(self.max_error_samples, len(error_indices))
        )
        num_errors = len(sampled_error_indices)

        # max allowed correct samples min(0.8 * num_errors, len(correct_indices), self.max_correct_samples)
        max_num_correct_samples = int(2 * max(1, num_errors))
        sampled_correct_indices = random.sample(
            correct_indices,
            min(
                self.max_correct_samples,
                max_num_correct_samples,
                len(correct_indices),
            ),
        )
        print(f"Subset Error size: {len(sampled_error_indices)}")
        print(f"Subset Correct size: {len(sampled_correct_indices)}")
        subset = sampled_error_indices + sampled_correct_indices
        # subset_samples = samples[sampled_error_indices + sampled_correct_indices]
        subset_score = np.mean(np.array(acc_score_list)[subset])
        print(f"Subset score: {subset_score}")

        return subset_score, subset

    def _track_effectiveness(
        self, stage: Literal["subset", "fullset", "valset", "demo_valset"], pass_: bool
    ):
        if stage == "subset":
            if pass_:
                self._subset_effect_count["pass"] += 1
            else:
                self._subset_effect_count["fail"] += 1
        elif stage == "fullset":
            if pass_:
                self._fullset_effect_count["pass"] += 1
            else:
                self._fullset_effect_count["fail"] += 1
        elif stage == "valset":
            if pass_:
                self._valset_effect_count["pass"] += 1
            else:
                self._valset_effect_count["fail"] += 1
        elif stage == "demo_valset":
            if pass_:
                self._demo_valset_effect_count["pass"] += 1
            else:
                self._demo_valset_effect_count["fail"] += 1
        else:
            raise NotImplementedError(f"Stage {stage} not implemented")

    def _text_grad_constraint_propose_step(
        self,
        current_step: int,
        all_samples,
        all_losses: List["Parameter"],
        all_y_preds,
        include_demo_optimizers: bool = False,
        trainer_results: TrainerResult = None,
        val_dataset: Any = None,
        test_dataset: Any = None,
    ):
        """Handles both the mixed training and the separate training.
        When include_demo_optimizers is True, the demo optimizers are included in the training
        """
        # comptute moving batch acc
        from adalflow.optim.parameter import Parameter

        for loss in all_losses:
            if not isinstance(loss, Parameter):
                raise ValueError("Loss should be a Parameter object")
        self.adaltask.eval()
        use_eval_loss_fn = False
        if self.adaltask.loss_eval_fn is not None:
            use_eval_loss_fn = True
        move_batch_eval = self.adaltask.evaluate_samples(
            all_samples, all_y_preds, use_loss_eval_fn=use_eval_loss_fn
        )
        print(f"Moving batch eval: {move_batch_eval}")
        move_batch_score = move_batch_eval.avg_score
        move_batch_acc_score_list = move_batch_eval.per_item_scores

        last_val_score = trainer_results.val_scores[-1]
        val_score_increased = False

        # if move_batch_score >= self.batch_val_score_threshold:
        #     print(f"Skipping batch {steps} as acc: {move_batch_score}")

        #     # reset the moving batch
        #     all_samples, all_losses, all_y_preds = [], [], []
        #     # track the result
        #     self._add_one_step_in_trainer_results(
        #         trainer_results,
        #         last_val_score,
        #         trainer_results.test_scores[-1],
        #         trainer_results.prompts[-1],
        #         total_steps,
        #     )
        #     return all_samples, all_losses, all_y_preds
        # downsample the moving batch
        all_samples, all_losses, all_y_preds, move_batch_acc_score_list = (
            self._downsample_move_batch(
                all_samples, all_losses, all_y_preds, move_batch_acc_score_list
            )
        )

        move_batch_score = np.mean(np.array(move_batch_acc_score_list))
        printc(f"Moving batch acc: {move_batch_score}")

        # create a subset with a more balanced error and correct samples
        subset_score, subset_indices = self._moving_batch_sample(
            move_batch_acc_score_list
        )
        printc(f"Subset batch acc: {subset_score},{subset_score}")

        self.adaltask.train()

        # compute the subset loss
        subset_losses = [all_losses[i] for i in subset_indices]

        subset_loss = sum_ops(subset_losses)
        print("Subset loss backward...")
        start_time = time.time()
        if self.disable_backward_gradients:
            self.adaltask.disable_backward_engine()

        if not self.disable_backward:  # no backward at all
            subset_loss.backward()
        print(f"Subset loss backward time: {time.time() - start_time}")  # 12seconds
        print("Optimizer propose...")
        # mark the subset loss to be backpropagated

        tdqm_loader = tqdm(range(self.max_proposals_per_step), desc="Proposing")

        for i in tdqm_loader:

            print(f"Proposal: {i+1}")
            start_time = time.time()
            self._propose_text_optimizers()  # new prompts
            printc(f"Propose time: {time.time()-start_time}")
            if include_demo_optimizers:
                self._demo_optimizers_propose()
            new_prompts = self.adaltask._get_param_values()
            print("New prompts: ", new_prompts)
            # valide the subset
            subset_samples = [all_samples[i] for i in subset_indices]
            val_output = self.adaltask.validation_step(
                subset_samples,
                current_step,
                self.num_workers,
                use_loss_eval_fn=use_eval_loss_fn,
            )
            # check subset validation score and compare with subset score
            val_score = val_output.avg_score
            if (
                val_score == subset_score
                and subset_score >= self.batch_val_score_threshold
            ) or val_score > subset_score:  # allow perfect subset to pass

                printc(
                    f"Pass minibatch check:{use_eval_loss_fn}, {val_score} > {subset_score}"
                )
                self._track_effectiveness("subset", True)

            else:
                printc(
                    f"Fail minibatch check, try next proposal: {use_eval_loss_fn}, {val_score} <= {subset_score}"
                )
                self._add_failed_proposals_text_optimizers()
                self._track_effectiveness("subset", False)
                self._revert_text_optimizers()
                if include_demo_optimizers:
                    self._demo_optimizers_revert()
                continue
            # validate the full set
            # move_batch_result = self.adaltask.validation_step(
            #     all_samples, steps, self.num_workers, use_loss_eval_fn=use_eval_loss_fn
            # )
            # new_move_batch_score = move_batch_result.avg_score
            # if new_move_batch_score >= move_batch_score:
            #     printc(f"Pass full check: {new_move_batch_score} >= {move_batch_score}")
            #     self._track_effectiveness("fullset", True)
            #     # break
            # else:
            #     printc(
            #         f"Fail full check, try next proposal: {new_move_batch_score} < {move_batch_score}"
            #     )
            #     self._track_effectiveness("fullset", False)
            #     self._add_failed_proposals_text_optimizers()
            #     self._revert_text_optimizers()
            #     if include_demo_optimizers:
            #         self._demo_optimizers_revert()
            #     continue

            # check on the validation set
            # set the batch size to the size of the validation set
            val_output = self.adaltask.validation_step(
                val_dataset,
                current_step,
                self.num_workers,
                minimum_score=last_val_score,
            )
            val_score = val_output.avg_score

            if val_score > last_val_score:
                print(f"Optimizer step: {val_score} > {last_val_score}")
                self._track_effectiveness("valset", True)
                self._step_text_optimizers()
                self._add_history_text_optimizers(val_score)

                if include_demo_optimizers:

                    self._demo_optimizers_step()

                # test the model
                test_score = None
                # if test_dataset is not None:
                #     test_output = self.adaltask.validation_step(
                #         test_dataset, total_steps, self.num_workers
                #     )
                #     test_score = test_output.avg_score

                self._add_one_step_in_trainer_results(
                    trainer_results,
                    val_score,
                    test_score,
                    new_prompts,
                    current_step,
                )
                all_samples, all_losses, all_y_preds = [], [], []
                val_score_increased = True
                self._reset_steps_from_last_improvement_text_optimizers()
                break
            else:
                print(f"Optimizer revert: {val_score} <= {last_val_score}")
                self._track_effectiveness("valset", False)
                self._add_failed_proposals_text_optimizers()
                self._revert_text_optimizers()
                if include_demo_optimizers:
                    self._demo_optimizers_revert()

                continue
        if not val_score_increased:
            print("No proposal can improve the subset and full set, and val set")
            self._zero_grad_text_optimizers()
            subset_loss.reset_all_gradients()
            # save the score, no change
            self._add_one_step_in_trainer_results(
                trainer_results,
                last_val_score,
                trainer_results.test_scores[-1],
                trainer_results.step_results[-1].prompt,
                current_step,
                attempted_val_score=val_score,
            )
            self._increment_step_from_last_improvement_text_optimizers()

        print(f"Saving checkpoint to {self.ckpt_file}")
        trainer_results.effective_measure = self._effective_measure
        save_json(trainer_results.to_dict(), self.ckpt_file)

        print("Done with proposals")
        self.adaltask.train()
        return all_samples, all_losses, all_y_preds

    def _fit_text_grad_constraint(
        self,
        train_loader: Any,
        val_dataset: Any,
        test_dataset: Any,
        trainer_results: TrainerResult = None,
        starting_step: int = 0,
    ) -> TrainerResult:
        """
        Starting_step != 0 when it is resume_from_ckpt
        """

        logger.info("Fitting using Textual Gradient Descent with constraints")
        printc("Fitting using Textual Gradient Descent with constraints")
        trainer_results = (
            self._pre_fit(val_dataset, test_dataset)
            if trainer_results is None
            else trainer_results
        )

        print(f"_fit_text_grad_constraint save to {self.ckpt_file}")

        self.adaltask.train()
        self._zero_grad_text_optimizers()

        num_epochs = self._estimate_num_epochs(train_loader, self.max_steps)
        current_step = starting_step
        all_samples, all_losses = [], []
        all_y_preds: List[OutputParameter] = []
        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            print(f"Epoch: {epoch}")
            for _, batch in enumerate((pbar := tqdm(train_loader, position=0))):
                current_step += 1
                if current_step > self.max_steps + starting_step:
                    print("Reached max steps")
                    break
                self._zero_grad_text_optimizers()
                self._text_optimizers_set_target_param()
                pbar.set_description(f"Training Step: {current_step}")
                self.adaltask.train()  # this will turn everything to train mode
                y_preds = self.adaltask.train_step(
                    batch, current_step, self.num_workers
                )
                losses = self.adaltask.loss_step(
                    batch, y_preds, current_step, self.num_workers
                )
                # moving batch

                all_samples.extend(batch)
                all_losses.extend(losses)
                all_y_preds.extend(
                    [y.data for y in y_preds if isinstance(y, OutputParameter)]
                )

                all_samples, all_losses, all_y_preds = (
                    self._text_grad_constraint_propose_step(
                        current_step=current_step,
                        all_samples=all_samples,
                        all_losses=all_losses,
                        all_y_preds=all_y_preds,
                        trainer_results=trainer_results,
                        val_dataset=val_dataset,
                        test_dataset=test_dataset,
                    )
                )

        save_json(trainer_results.to_dict(), self.ckpt_file)
        return trainer_results
