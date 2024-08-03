from typing import Any, Callable, Dict, List, Optional, Tuple
import concurrent
from tqdm import tqdm


from lightrag.core.component import Component
from lightrag.optim.optimizer import Optimizer
from lightrag.optim.types import PromptData
from lightrag.eval.base import BaseEvaluator, EvaluationResult


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

    def evaluate_samples(self, samples: Any, y_preds: List) -> EvaluationResult:
        r"""Run evaluation on samples.

        Note:
            ensure it supports both Tuple(batch) and a list of any type (fits for datasets).
        """
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
