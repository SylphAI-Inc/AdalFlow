"""Ready to use trainer for LLM task pipeline"""

from typing import Literal, Optional, List, Dict, Any, Tuple, TYPE_CHECKING
import os
import logging
from tqdm import tqdm
import random
import numpy as np

from lightrag.core.component import Component
from lightrag.optim.optimizer import Optimizer, DemoOptimizer, TextOptimizer

if TYPE_CHECKING:
    from lightrag.optim.parameter import Parameter
from lightrag.optim.types import PromptData, TrainerResult, FewShotConfig, ParameterType
from lightrag.eval.base import EvaluationResult
from lightrag.optim.trainer.adal import AdalComponent
from lightrag.optim.text_grad.ops import sum

from lightrag.utils import save_json
from lightrag.utils.cache import hash_text_sha1
from lightrag.utils.data import DataLoader


log = logging.getLogger(__name__)


class Trainer(Component):
    r"""We make trainer a component to as a trainer itself is an LLM task pipeline too.


    Training set: can be used for passing initial proposed prompt or for few-shot sampling.
    Validation set: Will be used to select the final prompt or samples.
    Test set: Will be used to evaluate the final prompt or samples.
    """

    adaltask: AdalComponent  # task pipeline
    # train_batch_size: int
    # train_dataset = (
    #     None  # accept library dataset  or pytorch dataset or huggingface dataset
    # )
    train_loader: Any
    val_dataset = None
    test_dataset = None
    # evaluator: object = None
    optimizer_type: Literal["text-grad", "orpo"] = "text-grad"
    strategy: Literal["random", "constrained"]
    max_steps: int
    optimizer: Optimizer = None
    ckpt_path: Optional[str] = None
    ckpt_file: Optional[str] = None
    num_workers: int = 2
    max_proposals_per_step: int = 5
    # moving batch for speed up the training
    batch_val_score_threshold: Optional[float] = (
        1.0  # when acc_score >= this threshold, skip this batch
    )
    max_error_samples: Optional[int] = 4
    max_correct_samples: Optional[int] = 4

    def __init__(
        self,
        adaltask: AdalComponent,
        optimizer_type: str = "text-grad",
        strategy: Literal["random", "constrained"] = "constrained",
        max_steps: int = 1000,
        num_workers: int = 2,
        ckpt_path: str = None,
        batch_val_score_threshold: Optional[float] = 1.0,
        max_error_samples: Optional[int] = 4,
        max_correct_samples: Optional[int] = 4,
        max_proposals_per_step: int = 5,
        train_loader: Optional[Any] = None,
        train_dataset: Optional[Any] = None,
        val_dataset: Optional[Any] = None,
        test_dataset: Optional[Any] = None,
        few_shots_config: Optional[FewShotConfig] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        if not isinstance(adaltask, AdalComponent):
            raise ValueError("Task should be an instance of AdalComponent")
        if strategy not in ["random", "constrained"]:
            raise ValueError("Strategy should be either random or constrained")
        self.optimizer_type = optimizer_type
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
        self.few_shots_config = few_shots_config

    def diagnose(self, train_dataset: Any):
        """Run an evaluation on the trainset to track all error response, and its raw response using AdaplComponent's default configure_callbacks

        Example:

        .. code-block:: python

            trainset, valset, testset = load_datasets(max_samples=10)
            adaltask = TGDWithEvalFnLoss(
                task_model_config=llama3_model,
                backward_engine_model_config=llama3_model,
                optimizer_model_config=llama3_model,
            )

            trainer = Trainer(adaltask=adaltask)
            diagnose = trainer.diagnose(train_dataset=trainset)
            print(diagnose)
        """
        # 1. track all intermediate outputs
        if not self.ckpt_path:
            trainer_state = self.gather_trainer_states()
            self.prep_ckpt_file_path(trainer_state)
        log_paths = self.adaltask.configure_callbacks(save_dir=self.ckpt_path)
        # 2. evaluate
        acc = self.adaltask.validation_step(train_dataset, 0, self.num_workers)
        acc_score = acc.avg_score
        acc_per_item_scores = acc.per_item_scores
        return acc_score, acc_per_item_scores, log_paths

    def fit(
        self,
        adaltask: Optional[AdalComponent] = None,
        train_loader: Optional[Any] = None,
        train_dataset: Optional[Any] = None,
        val_dataset: Optional[Any] = None,
        test_dataset: Optional[Any] = None,
        debug: bool = False,
    ):
        r"""
        train_loader: An iterable or collection of iterables specifying training samples.
        """

        train_loader = train_loader or self.train_loader

        train_dataset = train_dataset or self.train_dataset

        if not self.train_loader and train_dataset:
            batch_size = 4
            if self.strategy == "constrained":
                batch_size = 12
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
        adaltask = adaltask or self.adaltask

        if not isinstance(adaltask, AdalComponent):
            raise ValueError("Task should be an instance of AdalComponent")
        self.optimizers: List[Optimizer] = adaltask.configure_optimizers()
        self.text_optimizers = [
            opt for opt in self.optimizers if isinstance(opt, TextOptimizer)
        ]
        self.demo_optimizers = [
            opt for opt in self.optimizers if isinstance(opt, DemoOptimizer)
        ]
        if len(self.text_optimizers) > 0 or len(self.demo_optimizers) > 0:
            self.loss_fn = adaltask.configure_loss_fn()

        if debug:
            print("Debugging mode")
            if len(self.text_optimizers) > 0:
                self._fit_one_step_for_debug(self.train_loader)

            if len(self.demo_optimizers) > 0:
                adaltask.configure_teacher_generator()
                self._fit_demos_one_step_for_debug(
                    train_loader, train_dataset, val_dataset, test_dataset
                )
            return

        ########Run text_optimizers and demo optimizers in sequential order ########
        # TODO: check backward engine
        if len(self.text_optimizers) > 0:
            if self.strategy == "random":
                self._fit_text_grad_random(train_loader, val_dataset, test_dataset)
            elif self.strategy == "constrained":
                self._fit_text_grad_constraint(train_loader, val_dataset, test_dataset)
            else:
                raise ValueError(f"Strategy {self.strategy} not supported")

        # Run the demo optimizers
        if len(self.demo_optimizers) > 0:
            adaltask.configure_teacher_generator()
            pass

    @staticmethod
    def _estimate_num_epochs(train_loader: Any, max_steps: int):
        num_samples = len(train_loader)
        return max_steps // num_samples + 1

    def initial_validation(self, val_dataset: Any, test_dataset: Any):
        val_output = self.adaltask.validation_step(val_dataset, 0, self.num_workers)
        val_score = val_output.avg_score
        test_output = self.adaltask.validation_step(test_dataset, 0, self.num_workers)
        test_score = test_output.avg_score
        trainer_results = TrainerResult([], [], [], [])
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
        trainer_state["optimizer_type"] = self.optimizer_type
        trainer_state["strategy"] = self.strategy
        trainer_state["max_steps"] = self.max_steps
        trainer_state["num_workers"] = self.num_workers
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

        from lightrag.utils.serialization import serialize

        hash_key = hash_text_sha1(serialize(trainer_state))[0:5]
        trainer_state["hash_key"] = hash_key
        print(f"adal task hash key: {self.adaltask}")
        trainer_state["task_state_dict"] = self.adaltask.to_dict()
        restore_state = AdalComponent.from_dict(
            trainer_state["task_state_dict"]
        )  # tODO: add a test for adalcomponent
        print(
            f"restore_state: {str(restore_state.to_dict()) == str(self.adaltask.to_dict())}"
        )
        print(f"task_state_dict: {trainer_state['task_state_dict']}")
        return trainer_state

    def prep_ckpt_file_path(self, trainer_state: Dict[str, Any] = None):
        if self.ckpt_path is None:
            self.ckpt_path = os.path.join(
                os.getcwd(), "ckpt", self.adaltask.__class__.__name__
            )
            print(f"Checkpoint path: {self.ckpt_path}")
        os.makedirs(self.ckpt_path, exist_ok=True)
        # list all existing checkpoints with the same file name prefix
        file_name_prefix = (
            f"{self.strategy}_max_steps_{self.max_steps}_{trainer_state['hash_key']}"
        )
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

    def _pre_fit(self, val_dataset: Any, test_dataset: Any):
        # validate first (separate into another function where we can even save the outputs so that we can highlight error predictions)

        trainer_state = self.gather_trainer_states()
        trainer_results: TrainerResult = self.initial_validation(
            val_dataset, test_dataset
        )
        trainer_results.trainer_state = trainer_state
        self.prep_ckpt_file_path(trainer_state)
        return trainer_results
        # end of validation

    def _fit_one_step_for_debug(self, train_loader: Any):
        # reset batch size to 2
        train_loader.batch_size = 2
        batch = next(iter(train_loader))
        self.adaltask.train()  # this will turn everything to train mode
        y_preds = self.adaltask.train_step(batch, 0, self.num_workers)
        losses = self.adaltask.loss_step(batch, y_preds, 0, self.num_workers)
        total_loss = sum(losses)
        total_loss.backward()
        # test optimizer
        self.optimizer.propose()
        graph_path = os.path.join(self.ckpt_path, "graph.png")
        total_loss.draw_graph(filepath=graph_path)
        print(f"Graph saved to {graph_path}")

    def _set_demo_optimizers_dataset(self, train_dataset: Any):
        # init the dataset
        for opt in self.demo_optimizers:
            opt.sampler.set_dataset(train_dataset)

    def _demo_optimizers_propose(self):
        for opt in self.demo_optimizers:
            opt.propose()

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

    # TODO: make this the debug
    def _fit_demos_one_step_for_debug(
        self, train_loader, train_dataset: Any, val_dataset: Any, test_dataset: Any
    ):
        from lightrag.utils import get_logger

        get_logger(level="DEBUG")
        print("Fitting using Random Demo Optimizer")
        # trainer_results = self._pre_fit(val_dataset, test_dataset)
        print(f"save to {self.ckpt_file}")

        self.adaltask.train()
        self.adaltask.trace()
        self._set_demo_optimizers_dataset(train_dataset)
        # self.optimizer.zero_grad()

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
        print(f"Demo params: {demo_params}")

        if len(demo_params) == 0:
            raise ValueError("No demo params found")

        if len(demo_params[0]._traces) != 2:
            raise ValueError(f"Expected 2 traces, got {len(demo_params[0]._traces)}")

        print(f"Teacher y_preds: {y_preds[0].to_dict()}")

        # test validation mode

        batch_eval: EvaluationResult = self.adaltask.evaluate_samples(batch, y_preds)
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
        losses[0].backward()
        losses[1].backward()
        pred_teacher.add(batch[0].id)
        pred_teacher.add(batch[1].id)

        # check the score
        for key, val in demo_params[0]._traces.items():
            print(f"param: {key}, val: {val}")
            score = val.score
            if score is None:
                raise ValueError("Score is None")
            print(f"param: {key}, score: {score}")
        print(f"Loss after backward: {losses[0].to_dict()}")

        # test optimizer
        # self._init_demo_optimizers()
        # bootstrap
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
            losses_student: List[Parameter] = self.adaltask.loss_step(
                batch, y_preds_student, batch_idx, self.num_workers
            )
            for loss in losses_student:
                loss.backward()

            eval_result = self.adaltask.evaluate_samples(batch, y_preds_student)
            print(f"Eval result: {eval_result.avg_score}")
            eval_score_per_item = eval_result.per_item_scores

            # bootstrap
            batch_for_teacher = []

            for i, (sample, item_score) in enumerate(zip(batch, eval_score_per_item)):
                # use teacher
                if sample.id in pred_teacher:
                    continue
                if item_score < 0.5:
                    batch_for_teacher.append(sample)
                    pred_teacher.add(sample.id)
            # run teacher, use teachers's output instead of the initial output (bootstrap)
            if len(batch_for_teacher) > 0:
                self.adaltask.use_teacher()
                y_preds_teacher = self.adaltask.train_step(
                    batch_for_teacher, batch_idx, self.num_workers
                )
                losses_teacher: List[Parameter] = self.adaltask.loss_step(
                    batch_for_teacher, y_preds_teacher, batch_idx, self.num_workers
                )
                for loss in losses_teacher:
                    loss.backward()
            # propose
            self._demo_optimizers_propose()

            # validate
            # val_output = self.adaltask.validation_step(
            #     val_dataset, batch_idx, self.num_workers
            # )
            # val_score = val_output.avg_score
            # print(f"Validation score: {val_score}")  # 0.92 for two shot bootstrap

            # bootstrap
            # prefer to pick failed samples

            #

            # bootstrap
            # prefer to pick failed samples

        # print(f"Teacher raw y_preds: {y_preds[0].raw_response}")
        # print(f"Teacher input args: {y_preds[0].input_args}")
        # self._zero_grad_text_optimizers()

        # num_epochs = self._estimate_num_epochs(train_loader, self.max_steps)
        # total_steps = 0
        # for epoch in tqdm(range(num_epochs), desc="Epoch"):
        #     for steps, batch in enumerate((pbar := tqdm(train_loader, position=0))):
        #         total_steps += 1
        #         if total_steps > self.max_steps:
        #             print("Reached max steps")
        #             break
        #         pbar.set_description(f"Training Step: {total_steps}")
        #         self.adaltask.train()  # this will turn everything to train mode

    def _fit_text_grad_random(
        self, train_loader: Any, val_dataset: Any, test_dataset: Any
    ):
        log.info("Fitting using Textual Gradient Descent")
        trainer_results = self._pre_fit(val_dataset, test_dataset)
        print(f"save to {self.ckpt_file}")

        self.adaltask.train()
        # self.optimizer.zero_grad()
        self._zero_grad_text_optimizers()

        num_epochs = self._estimate_num_epochs(train_loader, self.max_steps)
        total_steps = 0
        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            for steps, batch in enumerate((pbar := tqdm(train_loader, position=0))):
                total_steps += 1
                if total_steps > self.max_steps:
                    print("Reached max steps")
                    break
                pbar.set_description(f"Training Step: {total_steps}")
                self.adaltask.train()  # this will turn everything to train mode
                y_preds = self.adaltask.train_step(batch, steps, self.num_workers)
                losses = self.adaltask.loss_step(
                    batch, y_preds, steps, self.num_workers
                )
                total_loss = sum(losses)
                print("Loss backward...")
                total_loss.backward()
                print("Optimizer propose...")
                self.optimizer.propose()
                new_prompts = self.adaltask._get_param_values()
                print("New prompts: ", new_prompts)
                if self.adaltask.validate_condition(steps, total_steps):
                    # set the batch size to the size of the validation set
                    val_output = self.adaltask.validation_step(
                        val_dataset,
                        steps,
                        self.num_workers,
                        minimum_score=trainer_results.val_scores[-1],
                    )
                    val_score = val_output.avg_score
                    if val_score > trainer_results.val_scores[-1]:
                        print(
                            f"Optimizer step: {val_score} > {trainer_results.val_scores[-1]}"
                        )
                        # self.optimizer.step()
                        self._step_text_optimizers()

                        # test the model
                        test_output = self.adaltask.validation_step(
                            test_dataset, steps, self.num_workers
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
                        print(
                            f"Optimizer revert: {val_score} <= {trainer_results.val_scores[-1]}"
                        )
                        # self.optimizer.revert()
                        self._revert_text_optimizers()
                        # save the score, no change
                        trainer_results.val_scores.append(
                            trainer_results.val_scores[-1]
                        )
                        trainer_results.test_scores.append(
                            trainer_results.test_scores[-1]
                        )
                        self._add_one_step_in_trainer_results(
                            trainer_results,
                            trainer_results.val_scores[-1],
                            trainer_results.test_scores[-1],
                            trainer_results.prompts[-1],
                            total_steps,
                        )

                print(f"Saving checkpoint to {self.ckpt_file}")
                save_json(trainer_results.to_dict(), self.ckpt_file)  # checkpoint

    @staticmethod
    def _add_one_step_in_trainer_results(
        trainer_results: TrainerResult,
        val_score: float,
        test_score: float,
        prompts: List[PromptData],
        steps: int,
    ):
        trainer_results.val_scores.append(val_score)
        trainer_results.test_scores.append(test_score)
        trainer_results.prompts.append(prompts)
        trainer_results.steps.append(steps)

    def _downsample_move_batch(
        self, all_samples, all_losses, all_y_preds, acc_score_list
    ):
        """Downsample the moving batch to a more balanced error and correct samples"""
        if not all([score in [0, 1] for score in acc_score_list]):
            raise ValueError("acc_score_list should only contain 0 and 1")
        max_moving_batch_size = 10

        correct_indices = [i for i, score in enumerate(acc_score_list) if score == 1]
        error_indices = [i for i, score in enumerate(acc_score_list) if score == 0]

        print(f"Moving batch correct size: {len(correct_indices)}")
        print(f"Moving batch error size: {len(error_indices)}")
        if (
            len(error_indices) <= max_moving_batch_size
            and len(correct_indices) <= max_moving_batch_size
        ):
            return all_samples, all_losses, all_y_preds, acc_score_list
        sampled_error_indices = correct_indices
        if len(correct_indices) > max_moving_batch_size:
            sampled_error_indices = random.sample(
                error_indices, min(max_moving_batch_size, len(error_indices))
            )
        sampled_correct_indices = correct_indices
        if len(correct_indices) > max_moving_batch_size:
            sampled_correct_indices = random.sample(
                correct_indices,
                min(
                    max_moving_batch_size,
                    len(correct_indices),
                ),
            )
        print(f"New moving batch size size: {len(sampled_error_indices)}")
        print(f"Subset Correct size: {len(sampled_correct_indices)}")
        new_sample_indices = sampled_error_indices + sampled_correct_indices
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
        max_num_correct_samples = 2 * num_errors
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
        all_losses,
        all_y_preds,
    ):
        # comptute moving batch acc
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
        subset_loss = sum(subset_losses)
        print("Subset loss backward...")
        subset_loss.backward()
        print("Optimizer propose...")

        # TODO: make this a step
        for i in range(self.max_proposals_per_step):
            print(f"Proposing step: {i}")
            # self.optimizer.propose()
            self._propose_text_optimizers()
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
                # self.optimizer.step()
            else:
                print(
                    f"Fail subset check, try next proposal: {val_score} <= {subset_score}"
                )
                self._track_effectiveness("subset", False)
                # self.optimizer.revert()
                self._revert_text_optimizers()
                continue  #
            # validate the full set
            move_batch_result = self.adaltask.validation_step(
                all_samples, steps, self.num_workers
            )
            new_move_batch_score = move_batch_result.avg_score
            if new_move_batch_score > move_batch_score:
                print(f"Pass full check: {new_move_batch_score} > {move_batch_score}")
                self._track_effectiveness("fullset", True)
                break
            else:
                print(
                    f"Fail full check, try next proposal: {new_move_batch_score} <= {move_batch_score}"
                )
                self._track_effectiveness("fullset", False)
                # self.optimizer.revert()
                self._revert_text_optimizers()
                continue

        print("Done with proposals")
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

    def _zero_grad_text_optimizers(self):
        for text_optimizer in self.text_optimizers:
            text_optimizer.zero_grad()

    def _propose_text_optimizers(self):
        for text_optimizer in self.text_optimizers:
            text_optimizer.propose()

    def _step_text_optimizers(self):
        for text_optimizer in self.text_optimizers:
            text_optimizer.step()

    def _revert_text_optimizers(self):
        for text_optimizer in self.text_optimizers:
            text_optimizer.revert()

    def _check_optimizer_proposal(self):
        r"""Return True if all optimizers have proposed a new prompt"""
        for text_optimizer in self.text_optimizers:
            if not text_optimizer.proposing:
                return False
        return True

    # TODO: miss one step somehow
    def _fit_text_grad_constraint(
        self, train_loader: Any, val_dataset: Any, test_dataset: Any
    ):
        log.info("Fitting using Textual Gradient Descent with constraints")
        trainer_results = self._pre_fit(val_dataset, test_dataset)

        print(f"save to {self.ckpt_file}")

        self.adaltask.train()
        self._zero_grad_text_optimizers()

        num_epochs = self._estimate_num_epochs(train_loader, self.max_steps)
        total_steps = 0
        all_samples, all_losses, all_y_preds = [], [], []
        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            for steps, batch in enumerate((pbar := tqdm(train_loader, position=0))):
                total_steps += 1
                if total_steps > self.max_steps:
                    print("Reached max steps")
                    break
                pbar.set_description(f"Training Step: {total_steps}")
                self.adaltask.train()  # this will turn everything to train mode
                y_preds = self.adaltask.train_step(batch, steps, self.num_workers)
                losses = self.adaltask.loss_step(
                    batch, y_preds, steps, self.num_workers
                )
                # moving batch

                all_samples.extend(batch)
                all_losses.extend(losses)
                all_y_preds.extend(y_preds)

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
                    val_output = self.adaltask.validation_step(
                        val_dataset,
                        steps,
                        self.num_workers,
                        minimum_score=trainer_results.val_scores[-1],
                    )
                    val_score = val_output.avg_score
                    if val_score > trainer_results.val_scores[-1]:
                        print(
                            f"Optimizer step: {val_score} > {trainer_results.val_scores[-1]}"
                        )
                        # self.optimizer.step()
                        self._step_text_optimizers()
                        # save the score
                        step_result = {
                            "val_score": val_score,
                        }

                        self._track_effectiveness("valset", True)

                        # test the model
                        test_output = self.adaltask.validation_step(
                            test_dataset,
                            steps,
                            self.num_workers,
                        )
                        step_result["test_score"] = test_output.avg_score
                        step_result["prompts"] = self.adaltask._get_param_values()
                        step_result["steps"] = total_steps
                        self._add_one_step_in_trainer_results(
                            trainer_results,
                            **step_result,
                        )
                        # test_score = test_output.avg_score
                        # trainer_results.test_scores.append(test_score)
                        # # save the prompts
                        # final_prompts = self.adaltask._get_param_values()
                        # trainer_results.prompts.append(final_prompts)
                        # reset the moving batch (the only difference from normal training)
                        all_samples, all_losses, all_y_preds = [], [], []

                    else:
                        print(
                            f"Optimizer revert: {val_score} <= {trainer_results.val_scores[-1]}"
                        )
                        # self.optimizer.revert()
                        self._revert_text_optimizers()
                        self._track_effectiveness("valset", False)
                        self._add_one_step_in_trainer_results(
                            trainer_results,
                            trainer_results.val_scores[-1],
                            trainer_results.test_scores[-1],
                            trainer_results.prompts[-1],
                            total_steps,
                        )

                trainer_results.effective_measure = self._effective_measure
                save_json(trainer_results.to_dict(), self.ckpt_file)  # checkpoint
        save_json(trainer_results.to_dict(), self.ckpt_file)  # checkpoint

    def validate(
        self, adaltask: Optional[AdalComponent], val_dataset: Any, max_samples: int
    ):
        r"""Perform one evaluation epoch over the validation set.

        val_loader: An iterable or collection of iterables specifying validation samples.
        """
        # wrap into a dataloader with large batch size, and control the number of samples
        pass

    def test(self, task: Optional[Component], test_loader: Any):
        r"""Perform one evaluation epoch over the test set. It's separated from fit to make sure you never run on your
        test set until you want to.

        test_loader: An iterable or collection of iterables specifying test samples.
        """
        pass

    def predict(self, task: Optional[Component], test_loader: Any):
        r"""Perform one evaluation epoch over the test set. It's separated from fit to make sure you never run on your
        test set until you want to.

        test_loader: An iterable or collection of iterables specifying test samples.
        """
        # load the best model from the checkpoint
        pass
