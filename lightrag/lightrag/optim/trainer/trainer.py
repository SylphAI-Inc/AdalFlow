"""Ready to use trainer for LLM task pipeline"""

from typing import Literal, Optional, List, Dict, Any, Callable, Tuple
import os
import logging
from tqdm import tqdm
import concurrent
from dataclasses import dataclass


from lightrag.core.component import Component
from lightrag.optim.text_grad.textual_grad_desc import TextualGradientDescent
from lightrag.optim.optimizer import Optimizer
from lightrag.optim.text_grad.ops import sum
from lightrag.eval.base import BaseEvaluator, EvaluationResult

from lightrag.utils import save_json
from lightrag.utils.cache import hash_text_sha1
from lightrag.core import DataClass


log = logging.getLogger(__name__)


@dataclass
class PromptData:
    id: str  # each parameter's id
    alias: str  # each parameter's alias
    data: str  # each parameter's data


class AdalComponent(Component):
    """Define a train, eval, and test step for a task pipeline.

    This serves the following purposes:
    1. Organize all parts for training a task pipeline organized in one place.
    2. Help with debugging and testing before the actual training.
    """

    task: Component
    evaluator: Optional[BaseEvaluator] = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _get_param_values(self) -> List[PromptData]:
        r"""Get the current values of the parameters."""
        return [PromptData(p.id, p.alias, p.data) for p in self.task.parameters()]
        # return {p.alias: p.data for p in self.task.parameters() if p.requires_opt}

    def handle_one_train_sample(self, sample: Any) -> Tuple[Callable, Dict]:
        r"""Parse the sample to the kwargs that the task pipeline can understand."""
        raise NotImplementedError("handle_one_train_sample method is not implemented")

    def handle_one_loss_sample(self, sample: Any, y_pred: Any) -> Tuple[Callable, Dict]:
        r"""Parse the sample to the kwargs that the task pipeline can understand."""
        raise NotImplementedError("handle_one_loss_sample method is not implemented")

    def run_one_train_sample(self, sample: Any) -> Any:
        r"""Run one training sample. Used for debugging and testing."""
        task_call, kwargs = self.handle_one_train_sample(sample)
        return task_call(**kwargs)

    def run_one_loss_sample(self, sample: Any, y_pred: Any) -> Any:
        r"""Run one loss sample. Used for debugging and testing."""
        loss_call, kwargs = self.handle_one_loss_sample(sample, y_pred)
        return loss_call(**kwargs)

    def evaluate_one_sample(self, sample: Any, y_pred: Any) -> Any:
        r"""Run one evaluation sample. Used for debugging and testing."""
        raise NotImplementedError("evaluate_one_sample method is not implemented")

    def evaluate_samples(self, samples: List, y_preds: List) -> EvaluationResult:
        r"""Run one evaluation sample. Used for debugging and testing."""
        raise NotImplementedError("evaluate_samples method is not implemented")

    def pred_step(
        self, batch, batch_idx, num_workers: int = 2, running_eval: bool = False
    ):
        r"""If you require self.task.train() to be called before training, you can override this method as:

        .. code-block:: python

            def train_step(self, batch, batch_idx, num_workers: int = 2) -> List:
                self.task.train()
                return super().train_step(batch, batch_idx, num_workers)
        """
        y_preds = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            tqdm_loader = tqdm(batch, total=len(batch), desc="Loading Data")
            for i, sample in enumerate(tqdm_loader):
                task_call, kwargs = self.handle_one_train_sample(sample)
                future = executor.submit(task_call, **kwargs)
                futures.append((future, sample))
            tqdm_loader = tqdm(
                futures,  # Pass only the futures to as_completed
                total=len(futures),
                position=0,
                desc="Evaluating",
            )
            samples = []
            for future, sample in tqdm_loader:
                # for future in futures:  # ensure the order of the predictions
                y_preds.append(future.result())
                samples.append(sample)
                if running_eval:
                    eval_score = self.evaluate_samples(samples, y_preds).avg_score
                    # TODO: compute max score to end evaluation early if it can not be improved
                    tqdm_loader.set_description(f"Evaluating: {eval_score}")
        # print("y_preds: ", y_preds)
        return y_preds

    def train_step(self, batch, batch_idx, num_workers: int = 2) -> List:
        raise NotImplementedError("train_step method is not implemented")

    def validate_condition(self, steps: int, total_steps: int) -> bool:
        r"""In default, trainer will validate at every step."""
        return True

    def validation_step(
        self, batch, batch_idx, num_workers: int = 2
    ) -> EvaluationResult:
        r"""If you require self.task.eval() to be called before validation, you can override this method as:

        .. code-block:: python

            def validation_step(self, batch, batch_idx, num_workers: int = 2) -> List:
                self.task.eval()
                return super().validation_step(batch, batch_idx, num_workers)
        """
        self.task.eval()
        y_preds = self.pred_step(batch, batch_idx, num_workers, running_eval=True)
        eval_results = self.evaluate_samples(batch, y_preds)
        print(f"eval_results: {eval_results}")
        return eval_results

    def loss_step(self, batch, y_preds, batch_idx, num_workers: int = 2):
        r"""Calculate the loss for the batch."""
        losses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            tqdm_loader = tqdm(
                zip(batch, y_preds), total=len(batch), desc="Loading Data"
            )
            for i, (sample, y_pred) in enumerate(tqdm_loader):
                loss_call, kwargs = self.handle_one_loss_sample(sample, y_pred)
                # losses.append(executor.submit(loss_call, **kwargs))
                futures.append(executor.submit(loss_call, **kwargs))
            tqdm_loader = tqdm(
                futures,  # Pass only the futures to as_completed
                total=len(futures),
                position=0,
                desc="Calculating Loss",
            )
            for future in tqdm_loader:
                losses.append(future.result())
        return losses

    def configure_optimizers(self, *args, **kwargs) -> Optimizer:
        raise NotImplementedError("configure_optimizers method is not implemented")

    def configure_backward_engine(self, *args, **kwargs):
        raise NotImplementedError("configure_backward_engine method is not implemented")

    def configure_loss_fn(self, *args, **kwargs):
        raise NotImplementedError("configure_loss_fn method is not implemented")


@dataclass
class TrainerResult(DataClass):
    steps: List[int]
    val_scores: List[float]
    test_scores: List[float]
    prompts: List[List[PromptData]]
    hyperparams: Dict[str, Any] = None


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
    # val_dataset = None
    # test_dataset = None
    # evaluator: object = None
    optimizer_type: Literal["text-grad", "orpo"] = "text-grad"
    strategy: Literal["Random", "constrained"]
    max_steps: int
    optimizer: Optimizer = None
    ckpt_path: Optional[str] = None
    ckpt_file: Optional[str] = None
    num_workers: int = 2

    def __init__(
        self,
        adaltask: AdalComponent,
        optimizer_type: str = "text-grad",
        strategy: Literal["Random", "constrained"] = "Random",
        max_steps: int = 1000,
        num_workers: int = 2,
        ckpt_path: str = None,
        train_loader: Optional[Any] = None,
        val_dataset: Optional[Any] = None,
        test_dataset: Optional[Any] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.optimizer_type = optimizer_type
        self.strategy = strategy
        self.max_steps = max_steps
        self.ckpt_path = ckpt_path
        self.adaltask = adaltask
        self.num_workers = num_workers
        self.train_loader = train_loader
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

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
        adaltask = adaltask or self.adaltask

        if not isinstance(adaltask, AdalComponent):
            raise ValueError("Task should be an instance of AdalComponent")
        self.optimizer: Optimizer = adaltask.configure_optimizers()
        if isinstance(self.optimizer, TextualGradientDescent):
            self.loss_fn = adaltask.configure_loss_fn()

        if self.optimizer_type == "text-grad":
            if self.strategy == "Random":
                self._fit_text_grad_random(train_loader, val_dataset, test_dataset)
            elif self.strategy == "constrained":
                pass
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

    def gather_all_hyperparams(self):
        hyperparams = {}
        hyperparams["optimizer_type"] = self.optimizer_type
        hyperparams["strategy"] = self.strategy
        hyperparams["max_steps"] = self.max_steps
        hyperparams["num_workers"] = self.num_workers
        hyperparams["batch_size"] = (
            self.train_loader.batch_size if self.train_loader else None
        )
        hyperparams["train_size"] = (
            len(self.train_loader.dataset) if self.train_loader else None
        )
        hyperparams["val_size"] = len(self.val_dataset) if self.val_dataset else None
        hyperparams["test_size"] = len(self.test_dataset) if self.test_dataset else None
        hyperparams["task_state_dict"] = self.adaltask.to_dict()
        restore_state = AdalComponent.from_dict(
            hyperparams["task_state_dict"]
        )  # tODO: add a test for adalcomponent
        print(
            f"restore_state: {str(restore_state.to_dict()) == str(self.adaltask.to_dict())}"
        )
        print(f"task_state_dict: {hyperparams['task_state_dict']}")
        from lightrag.utils.serialization import serialize

        hash_key = hash_text_sha1(serialize(hyperparams))[0:5]
        hyperparams["hash_key"] = hash_key
        return hyperparams

    def prep_ckpt_file_path(self, hyperparams: Dict[str, Any] = None):
        if self.ckpt_path is None:
            self.ckpt_path = os.path.join(
                os.getcwd(), "ckpt", self.adaltask.__class__.__name__
            )
            print(f"Checkpoint path: {self.ckpt_path}")
        os.makedirs(self.ckpt_path, exist_ok=True)
        # list all existing checkpoints with the same file name prefix
        file_name_prefix = (
            f"{self.strategy}_max_steps_{self.max_steps}_{hyperparams['hash_key']}"
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

    def _fit_text_grad_random(
        self, train_loader: Any, val_dataset: Any, test_dataset: Any
    ):
        log.info("Fitting using Textual Gradient Descent")
        # validate first (separate into another function where we can even save the outputs so that we can highlight error predictions)

        hyperparms = self.gather_all_hyperparams()
        trainer_results: TrainerResult = self.initial_validation(
            val_dataset, test_dataset
        )
        trainer_results.hyperparams = hyperparms
        self.prep_ckpt_file_path(hyperparms)
        # end of validation

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
