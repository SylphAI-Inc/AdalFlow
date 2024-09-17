"""Ready to use trainer for LLM task pipeline"""

from typing import Literal, Optional, List, Dict, Any, Tuple, TYPE_CHECKING
import os
import logging
from tqdm import tqdm
import random
import numpy as np
import uuid
import time

from adalflow.core.component import Component
from adalflow.optim.optimizer import Optimizer, DemoOptimizer, TextOptimizer

if TYPE_CHECKING:
    from adalflow.optim.parameter import Parameter
from adalflow.optim.types import (
    PromptData,
    TrainerResult,
    ParameterType,
    TrainerStepResult,
)
from adalflow.eval.base import EvaluationResult
from adalflow.optim.trainer.adal import AdalComponent
from adalflow.optim.text_grad.ops import sum_ops

from adalflow.utils import save_json, load_json
from adalflow.utils.cache import hash_text_sha1
from adalflow.utils.data import DataLoader

from adalflow.optim.types import TrainerValidateStats


log = logging.getLogger(__name__)


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
    max_error_samples: Optional[int] = 8
    max_correct_samples: Optional[int] = 8
    debug: bool = False

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
        max_error_samples: Optional[int] = 4,
        max_correct_samples: Optional[int] = 4,
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
        self.max_error_samples = max_error_samples
        self.max_correct_samples = max_correct_samples
        self.max_proposals_per_step = max_proposals_per_step

        self._subset_effect_count = {"pass": 0, "fail": 0}
        self._fullset_effect_count = {"pass": 0, "fail": 0}
        self._valset_effect_count = {"pass": 0, "fail": 0}
        self._effective_measure = {
            "subset": self._subset_effect_count,
            "fullset": self._fullset_effect_count,
            "valset": self._valset_effect_count,
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

    # TODO: need to support checkpoint resume too!
    def diagnose(self, dataset: Any, split: str = "train"):
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
        if not self.ckpt_path:
            trainer_state = self.gather_trainer_states()
            self.prep_ckpt_file_path(trainer_state)
        save_path = os.path.join(self.ckpt_path, f"diagnose_{split}")
        print(f"Save diagnose to {save_path}")
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
        print(f"sorted_indices: {sorted_indices}")
        sorted_scores = [acc_per_item_scores[i] for i in sorted_indices]
        print(f"sorted_scores: {sorted_scores}")
        sorted_dataset = [dataset[i] for i in sorted_indices]

        # reorder the samples based on the score
        for log_path in log_paths:
            file_name = os.path.basename(log_path)
            print(f"Loading log file: {file_name}")
            logs = load_jsonl(log_path)
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
            save_json(stats, os.path.join(log_dir, "stats.json"))
            print(f"Total error samples: {len(diagnose_items)}")
            print(f"Saved diagnose to {diagnose_file}")

        return acc_score, acc_per_item_scores, log_paths

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
    ):
        r"""
        train_loader: An iterable or collection of iterables specifying training samples.
        """
        start_time = time.time()

        debug = debug or self.debug
        if debug:
            from adalflow.utils import get_logger

            get_logger(level="DEBUG")

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

        if not train_loader and train_dataset:
            batch_size = self.train_batch_size

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
        val_dataset = val_dataset or self.val_dataset
        test_dataset = test_dataset or self.test_dataset
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
        self.optimizers: List[Optimizer] = self.adaltask.configure_optimizers()
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

        if len(self._get_trainable_text_params()) > 0:
            if self.adaltask.backward_engine is None:
                self.adaltask.configure_backward_engine()
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
            dict_data = load_json(self.ckpt_file)
            trainer_results: TrainerResult = TrainerResult.from_dict(dict_data)
            # restore the prompts to the adaltask
            val_scores = []
            test_scores = []
            for step in trainer_results.step_results:
                if step.val_score:
                    val_scores.append(step.val_score)
                if step.test_score:
                    test_scores.append(step.test_score)
            result_from_step = 0
            if test_scores:
                result_from_step = test_scores.index(max(test_scores))
            elif val_scores:
                result_from_step = val_scores.index(max(val_scores))
            prompts: List[PromptData] = trainer_results.step_results[
                result_from_step
            ].prompt

            print(f"Restoring prompts: {prompts[0]}")

            self.adaltask._set_param_values(prompts)
            starting_step = len(trainer_results.steps) - 1

        if debug:
            print("Debugging mode")
            # if len(self.text_optimizers) > 0:
            #     self._fit_text_grads_one_step_for_debug(train_loader)

            if len(self.demo_optimizers) > 0:
                self._fit_demos_one_step_for_debug(
                    train_loader, train_dataset, val_dataset, test_dataset
                )
            return

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
            if len(self.text_optimizers) > 0:
                if self.strategy == "random":
                    trainer_results = self._fit_text_grad_random(
                        train_loader,
                        val_dataset,
                        test_dataset,
                        trainer_results,
                        starting_step=starting_step,
                    )
                    starting_step += self.max_steps
                elif self.strategy == "constrained":
                    trainer_results = self._fit_text_grad_constraint(
                        train_loader,
                        val_dataset,
                        test_dataset,
                        trainer_results=trainer_results,
                        starting_step=starting_step,
                    )
                    starting_step += self.max_steps
                else:
                    raise ValueError(f"Strategy {self.strategy} not supported")
            if len(self.demo_optimizers) > 0:
                self.adaltask.configure_teacher_generator()  # attemp to use the newest teacher as
                self._fit_demos_random(
                    train_loader,
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    trainer_results=trainer_results,
                    starting_step=starting_step,
                )

        end_time = time.time()
        print(f"Training time: {end_time - start_time}s")
        print(f"ckpt_file: {self.ckpt_file}")

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
            steps=[], val_scores=[], test_scores=[], step_results=[], prompts=[]
        )
        trainer_results.val_scores.append(val_score)
        trainer_results.test_scores.append(test_score)
        prompts = self.adaltask._get_param_values()
        trainer_results.prompts.append(prompts)
        trainer_results.steps.append(0)
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
            print(f"Checkpoint path: {self.ckpt_path}")
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

    # TODO: make this the debug
    def _fit_demos_one_step_for_debug(
        self, train_loader, train_dataset: Any, val_dataset: Any, test_dataset: Any
    ):

        # get_logger(level="DEBUG")
        print("Fitting using Random Demo Optimizer")
        self.prep_ckpt_file_path()
        print(f"save to {self.ckpt_file}")

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

        print(f"Teacher y_preds: {y_preds[0].to_dict()}")

        y_preds_outputs = [p.full_response for p in y_preds]

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
        print(f"Losses: {losses[0].to_dict()}")
        self._demo_optimizers_add_scores(
            [sample.id for sample in batch], batch_per_item_scores, is_teacher=True
        )
        losses[0].backward()
        losses[1].backward()
        pred_teacher.add(batch[0].id)
        pred_teacher.add(batch[1].id)
        graph_path = os.path.join(self.ckpt_path, "graph")

        print(f"Graph saved to {graph_path}")

        # check the score
        for key, val in demo_params[0]._traces.items():
            print(f"param: {key}, val: {val}")
            score = val.score
            if score is None:
                raise ValueError("Score is None")
            print(f"param: {key}, score: {score}")
        print(f"Loss after backward: {losses[0].to_dict()}")

        # tracking the bootstrap so we wont repeat the same samples

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
            # for loss in losses_student:
            #     loss.backward()
            y_preds_outputs = [p.full_response for p in y_preds_student]
            eval_result = self.adaltask.evaluate_samples(batch, y_preds_outputs)
            print(f"Eval result: {eval_result.avg_score}")
            eval_score_per_item = eval_result.per_item_scores

            # bootstrap
            batch_for_teacher = []
            losses_teacher = []

            for i, (sample, item_score) in enumerate(zip(batch, eval_score_per_item)):
                # use teacher
                if sample.id in pred_teacher:
                    continue
                # if item_score < 0.5:
                batch_for_teacher.append(sample)
                pred_teacher.add(sample.id)
            # run teacher, use teachers's output instead of the initial output (bootstrap)
            if len(batch_for_teacher) > 0:
                print(f"Using teacher for {len(batch_for_teacher)} samples")
                self.adaltask.use_teacher()
                y_preds_teacher = self.adaltask.train_step(
                    batch_for_teacher, batch_idx, self.num_workers
                )
                losses_teacher: List[Parameter] = self.adaltask.loss_step(  # noqa F841
                    batch_for_teacher, y_preds_teacher, batch_idx, self.num_workers
                )
                self._demo_optimizers_add_scores(
                    [sample.id for sample in batch_for_teacher],
                    eval_score_per_item,
                    is_teacher=True,
                )

            # propose
            self._demo_optimizers_propose()
            graph_path = os.path.join(self.ckpt_path, "student_graph")

            losses_student[0].draw_graph(filepath=graph_path)

            # test step
            self._demo_optimizers_step()

            for opt in self.demo_optimizers:
                if opt.proposing:
                    raise ValueError("Optimizer is still proposing")
            # check demo params
            opt_params = []
            for opt in self.demo_optimizers:
                opt_params.extend(opt.params)
            print(f"Opt params: {opt_params}")
            for name, param in self.adaltask.named_parameters():
                print(f"Param: {name}")

                if param.param_type == ParameterType.DEMOS:
                    print(f"Demo param: {name}, value: {param.data}, param: {param}")
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

    def _fit_text_grads_one_step_for_debug(self, train_loader: Any):
        print("Debugging fitting one step with batch size 2 for text optimizer")
        from adalflow.utils import get_logger

        self.prep_ckpt_file_path()
        debug_path = os.path.join(self.ckpt_path, "debug_text_grads")
        os.makedirs(debug_path, exist_ok=True)
        get_logger(level="DEBUG", enable_console=False, save_dir=debug_path)
        train_loader.batch_size = 2
        train_loader.shuffle = True
        self.adaltask.train()  # this will turn everything to train mode
        correct_loss = None
        failed_loss = None
        print("Finding one successful and one failed loss")
        for batch in train_loader:
            y_preds = self.adaltask.train_step(batch, 0, self.num_workers)
            losses = self.adaltask.loss_step(batch, y_preds, 0, self.num_workers)
            for loss in losses:
                if loss.data > 0.5:
                    correct_loss = loss
                else:
                    failed_loss = loss
            if correct_loss and failed_loss:
                break
        total_loss = sum_ops([correct_loss, failed_loss])
        total_loss.backward()
        # test optimizer
        self._propose_text_optimizers()

        total_loss.draw_graph(filepath=debug_path)

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

    def _propose_text_optimizers(self):
        for text_optimizer in self.text_optimizers:
            text_optimizer.propose()

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

        log.info("Fitting using Textual Gradient Descent")
        trainer_results = (
            self._pre_fit(val_dataset, test_dataset)
            if trainer_results is None
            else trainer_results
        )
        print(f"save to {self.ckpt_file}")

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
                all_losses.extend(losses)
                # extract the non-parameter y_preds
                all_y_preds.extend(
                    [y.full_response for y in y_preds if isinstance(y, Parameter)]
                )

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
                    )
                )

                if not self._check_optimizer_proposal():
                    print(
                        "No proposal can improve the subset and full set, go to next step"
                    )

                    self._add_one_step_in_trainer_results(
                        trainer_results,
                        trainer_results.val_scores[-1],
                        trainer_results.test_scores[-1],
                        trainer_results.prompts[-1],
                        total_steps,
                    )
                    continue

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
                    if test_dataset is not None:
                        test_output = self.adaltask.validation_step(
                            test_dataset, total_steps, self.num_workers
                        )
                        test_score = test_output.avg_score

                    new_prompts = self.adaltask._get_param_values()
                    self._add_one_step_in_trainer_results(
                        trainer_results,
                        val_score,
                        test_score,
                        new_prompts,
                        total_steps,
                    )
                    all_samples, all_losses, all_y_preds = [], [], []
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
                        trainer_results.prompts[-1],
                        total_steps,
                        attempted_val_score=val_score,
                    )

                print(f"Saving checkpoint to {self.ckpt_file}")
                save_json(trainer_results.to_dict(), self.ckpt_file)
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
        log.info("Fitting using Textual Gradient Descent")

        trainer_results = (
            self._pre_fit(val_dataset, test_dataset)
            if train_results is None
            else train_results
        )
        print(f"save to {self.ckpt_file}")

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
                    test_output = self.adaltask.validation_step(
                        test_dataset, total_steps, self.num_workers
                    )
                    test_score = test_output.avg_score
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
                        trainer_results.prompts[-1],
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
        log.info("Fitting using Random Demo Optimizer")
        # self.adaltask.train()
        trainer_results = (
            self._pre_fit(val_dataset, test_dataset)
            if trainer_results is None
            else trainer_results
        )
        print(f"save to {self.ckpt_file}")
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
                loss.backward_engine_disabled = (
                    True  # temporary disable the backward engine
                )
                loss.backward()  # TODO: ensure no gradients in the backward, disable backward engine
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
                    self._demo_optimizers_step()
                    for opt in self.demo_optimizers:
                        if opt.proposing:
                            raise ValueError("Optimizer is still proposing")

                    # test the new prompts
                    test_score = None
                    if test_dataset is not None:
                        test_output = self.adaltask.validation_step(
                            test_dataset, step, self.num_workers
                        )
                        test_score = test_output.avg_score
                    self._add_one_step_in_trainer_results(
                        trainer_results,
                        val_score,
                        test_score=test_score,
                        prompts=new_prompts,
                        step=step,
                        attempted_val_score=val_score,
                    )
                else:
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
                        prompts=trainer_results.prompts[-1],
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

    def _fit_text_grad_random(
        self,
        train_loader: Any,
        val_dataset: Any,
        test_dataset: Any,
        trainer_results: TrainerResult = None,
        starting_step: int = 0,
    ) -> TrainerResult:
        log.info("Fitting using Textual Gradient Descent")
        trainer_results = (
            self._pre_fit(val_dataset, test_dataset)
            if trainer_results is None
            else trainer_results
        )
        print(f"save to {self.ckpt_file}")

        self.adaltask.train()
        # self.optimizer.zero_grad()
        self._zero_grad_text_optimizers()

        num_epochs = self._estimate_num_epochs(train_loader, self.max_steps)
        total_steps = starting_step
        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            for steps, batch in enumerate((pbar := tqdm(train_loader, position=0))):
                total_steps += 1
                if total_steps > self.max_steps + starting_step:
                    print("Reached max steps")
                    break
                self._zero_grad_text_optimizers()
                pbar.set_description(f"Training Step: {total_steps}")
                self.adaltask.train()  # this will turn everything to train mode
                self.train()
                y_preds = self.adaltask.train_step(batch, steps, self.num_workers)
                losses = self.adaltask.loss_step(
                    batch, y_preds, steps, self.num_workers
                )
                total_loss = sum_ops(losses)
                print("Loss backward...")
                total_loss.backward()
                print("Optimizer propose...")
                self._propose_text_optimizers()
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

                    # test the model
                    test_output = self.adaltask.validation_step(
                        test_dataset, total_steps, self.num_workers
                    )
                    test_score = test_output.avg_score
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
                    # save the score, no change
                    self._add_one_step_in_trainer_results(
                        trainer_results,
                        last_val_score,
                        trainer_results.test_scores[-1],
                        trainer_results.prompts[-1],
                        total_steps,
                        attempted_val_score=val_score,
                    )

                print(f"Saving checkpoint to {self.ckpt_file}")
                save_json(trainer_results.to_dict(), self.ckpt_file)
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
        trainer_results.prompts.append(prompts)
        trainer_results.steps.append(step)

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

        correct_indices = [i for i, score in enumerate(acc_score_list) if score > 0.5]
        error_indices = [i for i, score in enumerate(acc_score_list) if score <= 0.5]

        if (
            len(error_indices) + len(correct_indices)
            <= max_moving_batch_size
            # and len(correct_indices) <= max_moving_batch_size
        ):
            return all_samples, all_losses, all_y_preds, acc_score_list

        # downsample from all samples
        new_sample_indices = random.sample(
            range(len(all_samples)), min(max_moving_batch_size, len(all_samples))
        )
        all_samples = [all_samples[i] for i in new_sample_indices]
        all_losses = [all_losses[i] for i in new_sample_indices]
        all_y_preds = [all_y_preds[i] for i in new_sample_indices]
        acc_score_list = [acc_score_list[i] for i in new_sample_indices]
        return all_samples, all_losses, all_y_preds, acc_score_list

    def _moving_batch_sample(
        self, acc_score_list: List[float]
    ) -> Tuple[float, List[int]]:
        """Sample from both correct and error samples according to max_error_samples and max_correct_samples"""
        # ensure only 0 and 1 in the acc_score_list
        import numpy as np

        if not all([score in [0, 1] for score in acc_score_list]):
            raise ValueError("acc_score_list should only contain 0 and 1")
        correct_indices = [i for i, score in enumerate(acc_score_list) if score == 1]
        error_indices = [i for i, score in enumerate(acc_score_list) if score == 0]
        print(f"Moving batch correct size: {len(correct_indices)}")
        print(f"Moving batch error size: {len(error_indices)}")
        if len(error_indices) == 0:
            raise ValueError("No error samples found")
        sampled_error_indices = random.sample(
            error_indices, min(self.max_error_samples, len(error_indices))
        )
        num_errors = len(sampled_error_indices)

        # max allowed correct samples min(0.8 * num_errors, len(correct_indices), self.max_correct_samples)
        max_num_correct_samples = int(2 * num_errors)
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
        self, stage: Literal["subset", "fullset", "valset"], pass_: bool
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

    def _text_grad_constraint_propose_step(
        self,
        steps: int,
        all_samples,
        all_losses: List["Parameter"],
        all_y_preds,
        include_demo_optimizers: bool = False,
    ):
        # comptute moving batch acc
        from adalflow.optim.parameter import Parameter

        for loss in all_losses:
            if not isinstance(loss, Parameter):
                raise ValueError("Loss should be a Parameter object")
        self.adaltask.eval()
        move_batch_eval = self.adaltask.evaluate_samples(all_samples, all_y_preds)
        move_batch_score = move_batch_eval.avg_score
        move_batch_acc_score_list = move_batch_eval.per_item_scores

        if move_batch_score >= self.batch_val_score_threshold:
            print(f"Skipping batch {steps} as acc: {move_batch_score}")

            # reset the moving batch
            all_samples, all_losses, all_y_preds = [], [], []
            return all_samples, all_losses, all_y_preds
        # downsample the moving batch
        all_samples, all_losses, all_y_preds, move_batch_acc_score_list = (
            self._downsample_move_batch(
                all_samples, all_losses, all_y_preds, move_batch_acc_score_list
            )
        )

        move_batch_score = np.mean(np.array(move_batch_acc_score_list))
        print(f"Moving batch acc: {move_batch_score}")

        # create a subset with a more balanced error and correct samples
        subset_score, subset_indices = self._moving_batch_sample(
            move_batch_acc_score_list
        )
        print(f"Subset batch acc: {subset_score}")

        # compute the subset loss
        subset_losses = [all_losses[i] for i in subset_indices]

        subset_loss = sum_ops(subset_losses)
        print("Subset loss backward...")
        start_time = time.time()
        subset_loss.backward()
        print(f"Subset loss backward time: {time.time() - start_time}")  # 12seconds
        print("Optimizer propose...")
        # mark the subset loss to be backpropagated

        # TODO: make this a step
        tdqm_loader = tqdm(range(self.max_proposals_per_step), desc="Proposing")
        for i in tdqm_loader:

            # print(f"Proposing step: {i}")
            # self.optimizer.propose()
            self._propose_text_optimizers()  # new prompts
            if include_demo_optimizers:
                self._demo_optimizers_propose()
            new_prompts = self.adaltask._get_param_values()
            print("New prompts: ", new_prompts)
            # valide the subset
            subset_samples = [all_samples[i] for i in subset_indices]
            # validate the subset
            val_output = self.adaltask.validation_step(
                subset_samples, steps, self.num_workers
            )
            # check subset validation score
            val_score = val_output.avg_score
            if val_score > subset_score:
                print(f"Pass subset check: {val_score} > {subset_score}")
                self._track_effectiveness("subset", True)

            else:
                print(
                    f"Fail subset check, try next proposal: {val_score} <= {subset_score}"
                )
                self._track_effectiveness("subset", False)
                self._revert_text_optimizers()
                if include_demo_optimizers:
                    self._demo_optimizers_revert()
                continue
            # validate the full set
            move_batch_result = self.adaltask.validation_step(
                all_samples, steps, self.num_workers
            )
            new_move_batch_score = move_batch_result.avg_score
            if new_move_batch_score >= move_batch_score:
                print(f"Pass full check: {new_move_batch_score} >= {move_batch_score}")
                self._track_effectiveness("fullset", True)
                break
            else:
                print(
                    f"Fail full check, try next proposal: {new_move_batch_score} < {move_batch_score}"
                )
                self._track_effectiveness("fullset", False)
                self._revert_text_optimizers()
                if include_demo_optimizers:
                    self._demo_optimizers_revert()
                continue

        print("Done with proposals")
        self.adaltask.train()
        return all_samples, all_losses, all_y_preds

    # def _fit_bootstrap_few_shot_random(
    #     self,
    #     train_loader: Any,
    #     val_dataset: Any,
    #     test_dataset: Any,
    #     optimizers: List[DemoOptimizer],
    # ):
    #     log.info("Fitting using Bootstrap Few Shot only")
    #     trainer_results = self._pre_fit(val_dataset, test_dataset)
    #     print(f"save to {self.ckpt_file}")

    #     self.adaltask.train()  #

    #     num_epochs = self._estimate_num_epochs(train_loader, self.max_steps)
    #     total_steps = 0
    #     for optimizer in optimizers:
    #         optimizer.init()
    #     for epoch in tqdm(range(num_epochs), desc="Epoch"):
    #         for steps, batch in enumerate((pbar := tqdm(train_loader, position=0))):
    #             total_steps += 1
    #             if total_steps > self.max_steps:
    #                 print("Reached max steps")
    #                 break
    #             pbar.set_description(f"Training Step: {total_steps}")
    #             self.adaltask.train()

    def _fit_text_grad_constraint(
        self,
        train_loader: Any,
        val_dataset: Any,
        test_dataset: Any,
        trainer_results: TrainerResult = None,
        starting_step: int = 0,
    ) -> TrainerResult:
        from adalflow.optim.parameter import Parameter

        log.info("Fitting using Textual Gradient Descent with constraints")
        trainer_results = (
            self._pre_fit(val_dataset, test_dataset)
            if trainer_results is None
            else trainer_results
        )

        print(f"save to {self.ckpt_file}")

        self.adaltask.train()
        self._zero_grad_text_optimizers()

        num_epochs = self._estimate_num_epochs(train_loader, self.max_steps)
        total_steps = starting_step
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
                y_preds = self.adaltask.train_step(batch, steps, self.num_workers)
                losses = self.adaltask.loss_step(
                    batch, y_preds, steps, self.num_workers
                )
                # moving batch

                all_samples.extend(batch)
                all_losses.extend(losses)
                all_y_preds.extend(
                    [y.full_response for y in y_preds if isinstance(y, Parameter)]
                )

                all_samples, all_losses, all_y_preds = (
                    self._text_grad_constraint_propose_step(
                        steps=steps,
                        all_samples=all_samples,
                        all_losses=all_losses,
                        all_y_preds=all_y_preds,
                    )
                )

                # check optimizer stages to see if the proposal was accepted so far
                if not self._check_optimizer_proposal():
                    print(
                        "No proposal can improve the subset and full set, go to next step"
                    )

                    self._add_one_step_in_trainer_results(
                        trainer_results,
                        trainer_results.val_scores[-1],
                        trainer_results.test_scores[-1],
                        trainer_results.prompts[-1],
                        total_steps,
                    )
                    continue

                # prune the correct sample size if its too big, same with error samples
                # run the tests as any other optimizer
                if self.adaltask.validate_condition(steps, total_steps):
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

                        # save the score
                        step_result = {
                            "val_score": val_score,
                        }

                        self._track_effectiveness("valset", True)

                        # test the model
                        if test_dataset is not None:
                            test_output = self.adaltask.validation_step(
                                test_dataset,
                                steps,
                                self.num_workers,
                            )
                            step_result["test_score"] = test_output.avg_score
                        else:
                            step_result["test_score"] = None
                        step_result["prompts"] = self.adaltask._get_param_values()
                        step_result["step"] = total_steps
                        self._add_one_step_in_trainer_results(
                            trainer_results,
                            **step_result,
                        )

                        all_samples, all_losses, all_y_preds = [], [], []

                    else:
                        print(f"Optimizer revert: {val_score} <= {last_val_score}")
                        self._revert_text_optimizers()
                        self._track_effectiveness("valset", False)
                        self._add_one_step_in_trainer_results(
                            trainer_results,
                            trainer_results.val_scores[-1],
                            trainer_results.test_scores[-1],
                            trainer_results.prompts[-1],
                            total_steps,
                            attempted_val_score=val_score,
                        )

                trainer_results.effective_measure = self._effective_measure
                save_json(trainer_results.to_dict(), self.ckpt_file)
        save_json(trainer_results.to_dict(), self.ckpt_file)
        return trainer_results
