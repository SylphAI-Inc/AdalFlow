"""Ready to use trainer for LLM task pipeline"""

from typing import Literal, Optional, List, Dict, Any, Tuple
import os
import logging
from tqdm import tqdm
import random

from lightrag.core.component import Component
from lightrag.optim.text_grad.textual_grad_desc import TextualGradientDescent
from lightrag.optim.optimizer import Optimizer
from lightrag.optim.types import PromptData, TrainerResult
from lightrag.optim.trainer.adal import AdalComponent
from lightrag.optim.text_grad.ops import sum

from lightrag.utils import save_json
from lightrag.utils.cache import hash_text_sha1


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
        val_dataset: Optional[Any] = None,
        test_dataset: Optional[Any] = None,
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
        val_dataset: Optional[Any] = None,
        test_dataset: Optional[Any] = None,
    ):
        r"""
        train_loader: An iterable or collection of iterables specifying training samples.
        """

        self.train_loader = train_loader or self.train_loader
        self.val_dataset = val_dataset or self.val_dataset
        self.test_dataset = test_dataset or self.test_dataset
        # check train_loader and val_dataset and test_dataset, reject tuple
        if self.train_loader:
            exam_batch = next(iter(self.train_loader))
            if isinstance(exam_batch, tuple):
                raise ValueError(
                    "train_loader should return not be tuple, please use dict or a dataclass or with DataClass"
                )
        if self.val_dataset:
            if isinstance(self.val_dataset, tuple):
                raise ValueError(
                    "val_dataset should not be tuple, please use dict or a dataclass or with DataClass"
                )
        if self.test_dataset:
            if isinstance(self.test_dataset, tuple):
                raise ValueError(
                    "test_dataset should not be tuple, please use dict or a dataclass or with DataClass"
                )
        adaltask = adaltask or self.adaltask

        if not isinstance(adaltask, AdalComponent):
            raise ValueError("Task should be an instance of AdalComponent")
        self.optimizer: Optimizer = adaltask.configure_optimizers()
        if isinstance(self.optimizer, TextualGradientDescent):
            self.loss_fn = adaltask.configure_loss_fn()

        if self.optimizer_type == "text-grad":
            if self.strategy == "random":
                self._fit_text_grad_random(train_loader, val_dataset, test_dataset)
            elif self.strategy == "constrained":
                self._fit_text_grad_constraint(train_loader, val_dataset, test_dataset)
        elif self.optimizer_type == "orpo":
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

    def _fit_text_grad_random(
        self, train_loader: Any, val_dataset: Any, test_dataset: Any
    ):
        log.info("Fitting using Textual Gradient Descent")
        trainer_results = self._pre_fit(val_dataset, test_dataset)

        self.adaltask.train()
        self.optimizer.zero_grad()

        num_epochs = self._estimate_num_epochs(train_loader, self.max_steps)
        total_steps = 0
        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            for steps, (batch_x, batch_y) in enumerate(
                (pbar := tqdm(train_loader, position=0))
            ):
                total_steps += 1
                if total_steps > self.max_steps:
                    print("Reached max steps")
                    break
                pbar.set_description(f"Training Step: {total_steps}")
                self.adaltask.train()  # this will turn everything to train mode
                y_preds = self.adaltask.train_step(batch_x, steps, self.num_workers)
                losses = self.adaltask.loss_step(
                    batch_x, y_preds, steps, self.num_workers
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
                        val_dataset, steps, self.num_workers
                    )
                    val_score = val_output.avg_score
                    if val_score > trainer_results.val_scores[-1]:
                        print(
                            f"Optimizer step: {val_score} > {trainer_results.val_scores[-1]}"
                        )
                        self.optimizer.step()
                        # save the score
                        trainer_results.val_scores.append(val_score)

                        # test the model
                        test_output = self.adaltask.validation_step(
                            test_dataset, steps, self.num_workers
                        )
                        test_score = test_output.avg_score
                        trainer_results.test_scores.append(test_score)
                    else:
                        print(
                            f"Optimizer revert: {val_score} <= {trainer_results.val_scores[-1]}"
                        )
                        self.optimizer.revert()
                        # save the score, no change
                        trainer_results.val_scores.append(
                            trainer_results.val_scores[-1]
                        )
                        trainer_results.test_scores.append(
                            trainer_results.test_scores[-1]
                        )
                #
                # save the results
                final_prompts = self.adaltask._get_param_values()
                trainer_results.prompts.append(final_prompts)
                trainer_results.steps.append(total_steps)
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
        # return {
        #     "subset": self._subset_effect_count,
        #     "fullset": self._fullset_effect_count,
        #     "valset": self._valset_effect_count,
        # }

    def _text_grad_constraint_propose_step(
        self,
        steps: int,
        trainer_results: TrainerResult,
        all_samples,
        all_losses,
        all_y_preds,
    ):
        # comptute moving batch acc
        # self.adaltask.eval()
        move_batch_eval = self.adaltask.evaluate_samples(all_samples, all_y_preds)
        move_batch_score = move_batch_eval.avg_score
        move_batch_acc_score_list = move_batch_eval.per_item_scores
        print(f"Moving batch acc: {move_batch_score}")
        if move_batch_score >= self.batch_val_score_threshold:
            print(f"Skipping batch {steps} as acc: {move_batch_score}")
            # self._add_one_step_in_trainer_results(
            #     trainer_results,
            #     trainer_results.val_scores[-1],
            #     trainer_results.test_scores[-1],
            #     trainer_results.prompts[-1],
            #     steps,
            # )
            # reset the moving batch
            all_samples, all_losses, all_y_preds = [], [], []
            return
        # downsample the moving batch
        all_samples, all_losses, all_y_preds, move_batch_acc_score_list = (
            self._downsample_move_batch(
                all_samples, all_losses, all_y_preds, move_batch_acc_score_list
            )
        )

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
            self.optimizer.propose()
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
                self.optimizer.revert()
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
                self.optimizer.revert()
                continue

        print("Done with proposals")

    # TODO: miss one step somehow
    def _fit_text_grad_constraint(
        self, train_loader: Any, val_dataset: Any, test_dataset: Any
    ):
        log.info("Fitting using Textual Gradient Descent with constraints")
        trainer_results = self._pre_fit(val_dataset, test_dataset)

        print(f"save to {self.ckpt_file}")

        self.adaltask.train()
        self.optimizer.zero_grad()

        self.adaltask.train()
        self.optimizer.zero_grad()

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

                self._text_grad_constraint_propose_step(
                    steps=steps,
                    trainer_results=trainer_results,
                    all_samples=all_samples,
                    all_losses=all_losses,
                    all_y_preds=all_y_preds,
                )

                # check optimizer stages to see if the proposal was accepted so far
                if not self.optimizer.proposing:
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
                        self.optimizer.step()
                        # save the score
                        step_result = {
                            "val_score": val_score,
                        }
                        # trainer_results.val_scores.append(val_score)

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
                        self.optimizer.revert()
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
