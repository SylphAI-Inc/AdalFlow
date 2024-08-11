"""AdalComponent provides an interface to compose different parts, from eval_fn, train_step, loss_step, optimizers, backward engine, teacher generator, etc to work with Trainer."""

from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
import concurrent
from tqdm import tqdm
import numpy as np
import warnings

if TYPE_CHECKING:
    from adalflow.core.model_client import ModelClient
    from adalflow.core.generator import Generator, BackwardEngine
    from adalflow.optim.parameter import Parameter

from adalflow.core.component import Component
from adalflow.optim.optimizer import Optimizer
from adalflow.optim.loss_component import LossComponent
from adalflow.optim.types import PromptData
from adalflow.eval.base import BaseEvaluator, EvaluationResult

from adalflow.optim.optimizer import DemoOptimizer, TextOptimizer


# TODO: test step
class AdalComponent(Component):
    """Define a train, eval, and test step for a task pipeline.

    This serves the following purposes:
    1. Organize all parts for training a task pipeline organized in one place.
    2. Help with debugging and testing before the actual training.
    """

    task: Component
    evaluator: Optional[BaseEvaluator]
    eval_fn: Optional[Callable]
    loss_fn: Optional[LossComponent]
    backward_engine: Optional["BackwardEngine"]

    def __init__(
        self,
        task: Component,
        evaluator: Optional[BaseEvaluator] = None,
        eval_fn: Optional[Callable] = None,
        loss_fn: Optional[LossComponent] = None,
        backward_engine: Optional["BackwardEngine"] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.task = task
        self.evaluator = evaluator
        self.eval_fn = eval_fn
        self.loss_fn = loss_fn
        self.backward_engine = backward_engine

    def _get_param_values(self) -> List[PromptData]:
        r"""Get the current values of the parameters."""
        return [PromptData(p.id, p.alias, p.data) for p in self.task.parameters()]
        # return {p.alias: p.data for p in self.task.parameters() if p.requires_opt}

    def handle_one_task_sample(
        self, sample: Any, *args, **kwargs
    ) -> Tuple[Callable, Dict]:
        r"""Return a task call and kwargs for one training sample.

        Example:

        .. code-block:: python

            def handle_one_task_sample(self, sample: Any, *args, **kwargs) -> Tuple[Callable, Dict]:
                return self.task, {"x": sample.x}
        """

        raise NotImplementedError("handle_one_task_sample method is not implemented")

    def handle_one_loss_sample(
        self, sample: Any, y_pred: "Parameter", *args, **kwargs
    ) -> Tuple[Callable, Dict]:
        r"""Return a loss call and kwargs for one loss sample.

        Example:

        .. code-block:: python

            def handle_one_loss_sample(self, sample: Any, y_pred: Any, *args, **kwargs) -> Tuple[Callable, Dict]:
                return loss_fn, {"y_pred": y_pred, "y": sample.y}
        """
        raise NotImplementedError("handle_one_loss_sample method is not implemented")

    # TODO: support more complicated evaluation
    def evaluate_one_sample(self, sample: Any, y_pred: Any, *args, **kwargs) -> float:
        r"""Used to evaluate a single sample. Return a score in range [0, 1].
        The higher the score the better the prediction."""
        raise NotImplementedError("evaluate_one_sample method is not implemented")

    def configure_optimizers(self, *args, **kwargs) -> Optimizer:
        r"""Note: When you use text optimizor, ensure you call `configure_backward_engine_engine` too."""
        raise NotImplementedError("configure_optimizers method is not implemented")

    def configure_backward_engine(self, *args, **kwargs):
        raise NotImplementedError("configure_backward_engine method is not implemented")

    # def configure_loss_fn(self, *args, **kwargs):
    #     raise NotImplementedError("configure_loss_fn method is not implemented")

    def evaluate_samples(
        self, samples: Any, y_preds: List, metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        r"""Run evaluation on samples.

        Metadata is used for storing context that you can find from generator input.

        Note:
            ensure it supports both Tuple(batch) and a list of any type (fits for datasets).
        """
        # in default use evaluate_one_sample
        acc_list = [
            self.evaluate_one_sample(sample, y_pred, metadata=metadata)
            for sample, y_pred in zip(samples, y_preds)
        ]
        avg_score = np.mean(np.array(acc_list))
        return EvaluationResult(avg_score=avg_score, per_item_scores=acc_list)
        # raise NotImplementedError("evaluate_samples method is not implemented")

    def pred_step(
        self,
        batch,
        batch_idx,
        num_workers: int = 2,
        running_eval: bool = False,
        min_score: Optional[float] = None,
    ):
        r"""If you require self.task.train() to be called before training, you can override this method as:

        .. code-block:: python

            def train_step(self, batch, batch_idx, num_workers: int = 2) -> List:
                self.task.train()
                return super().train_step(batch, batch_idx, num_workers)
        """
        from adalflow.optim.parameter import Parameter

        y_preds = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            tqdm_loader = tqdm(batch, total=len(batch), desc="Loading Data")
            for i, sample in enumerate(tqdm_loader):
                task_call, kwargs = self.handle_one_task_sample(sample)
                future = executor.submit(task_call, **kwargs)
                futures.append((future, sample))
            tqdm_loader = tqdm(
                futures,  # Pass only the futures to as_completed
                total=len(futures),
                position=0,
                desc="Evaluating",
            )
            samples = []
            for i, (future, sample) in enumerate(tqdm_loader):
                # for future in futures:  # ensure the order of the predictions
                y_pred = future.result()
                y_preds.append(future.result())
                samples.append(sample)
                if running_eval and not isinstance(y_pred, Parameter):
                    remain_samples = len(futures) - (i + 1)
                    eval_score = self.evaluate_samples(samples, y_preds).avg_score
                    max_score = (eval_score * (i + 1) + remain_samples) / len(futures)
                    if min_score is not None and max_score < min_score:
                        break
                    # TODO: compute max score to end evaluation early if it can not be improved
                    tqdm_loader.set_description(
                        f"Evaluating: {eval_score}, Max potential: {max_score}"
                    )
                else:
                    tqdm_loader.set_description("Evaluating")
        return y_preds

    # def train_step(self, batch, batch_idx, num_workers: int = 2) -> List:
    #     raise NotImplementedError("train_step method is not implemented")

    def train_step(self, batch, batch_idx, num_workers: int = 2) -> List:
        self.task.train()
        y_preds = self.pred_step(batch, batch_idx, num_workers)
        for i, y_pred in enumerate(y_preds):
            try:
                y_pred.alias += f"y_pred_{i}"
            except AttributeError:
                raise ValueError(f"y_pred_{i} is not a Parameter, {y_pred}")
        return y_preds

    def validate_condition(self, steps: int, total_steps: int) -> bool:
        r"""In default, trainer will validate at every step."""
        return True

    def validation_step(
        self,
        batch,
        batch_idx,
        num_workers: int = 2,
        minimum_score: Optional[float] = None,
    ) -> EvaluationResult:
        r"""If you require self.task.eval() to be called before validation, you can override this method as:

        .. code-block:: python

            def validation_step(self, batch, batch_idx, num_workers: int = 2) -> List:
                self.task.eval()
                return super().validation_step(batch, batch_idx, num_workers)
        """
        # TODO: let use decide which mode to be
        self.task.eval()
        y_preds = self.pred_step(
            batch, batch_idx, num_workers, running_eval=True, min_score=minimum_score
        )
        eval_results = self.evaluate_samples(batch, y_preds)
        return eval_results

    def loss_step(
        self, batch, y_preds: List["Parameter"], batch_idx, num_workers: int = 2
    ) -> List["Parameter"]:
        r"""Calculate the loss for the batch."""
        from adalflow.optim.parameter import Parameter

        losses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            tqdm_loader = tqdm(
                zip(batch, y_preds), total=len(batch), desc="Loading Data"
            )
            for i, (sample, y_pred) in enumerate(tqdm_loader):
                loss_forward, kwargs = self.handle_one_loss_sample(sample, y_pred)
                # losses.append(executor.submit(loss_call, **kwargs))
                futures.append(executor.submit(loss_forward, **kwargs))
            tqdm_loader = tqdm(
                futures,  # Pass only the futures to as_completed
                total=len(futures),
                position=0,
                desc="Calculating Loss",
            )
            for future in tqdm_loader:
                loss = future.result()
                if not isinstance(loss, Parameter):
                    raise ValueError(f"Loss is not a Parameter: {loss}")
                losses.append(loss)
        return losses

    def configure_teacher_generator(self):
        r"""Configure a teach generator for all generators in the task for bootstrapping examples.

        You can call `configure_teacher_generator_helper` to easily configure it by passing the model_client and model_kwargs.
        """
        raise NotImplementedError(
            "configure_teacher_generator method is not implemented"
        )

    def configure_backward_engine_helper(
        self,
        model_client: "ModelClient",
        model_kwargs: Dict[str, Any],
        template: Optional[str] = None,
    ):
        r"""Configure a backward engine for all generators in the task for bootstrapping examples."""
        from adalflow.core.generator import BackwardEngine

        self.backward_engine = BackwardEngine(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=template,
        )

        # set all generator's backward engine

        all_generators = self._find_all_generators()
        for _, generator in all_generators:
            generator.set_backward_engine(self.backward_engine)
        print("Backward engine configured for all generators.")

        if not self.loss_fn:
            raise ValueError("Loss function is not configured.")

        # configure it for loss_fn
        if self.loss_fn:
            self.loss_fn.set_backward_engine(self.backward_engine)

    def configure_teacher_generator_helper(
        self,
        model_client: "ModelClient",
        model_kwargs: Dict[str, Any],
        template: Optional[str] = None,
    ):
        r"""Configure a teach generator for all generators in the task for bootstrapping examples."""
        from adalflow.core.generator import create_teacher_generator

        all_generators = self._find_all_generators()
        for _, generator in all_generators:
            teacher_generator = create_teacher_generator(
                student=generator,
                model_client=model_client,
                model_kwargs=model_kwargs,
                template=template,
            )
            print(f"Configuring teacher generator for {teacher_generator}")
            generator.set_teacher_generator(teacher_generator)
        print("Teacher generator configured.")

    def configure_callbacks(self, save_dir: str = "traces", *args, **kwargs):
        """In default we config the failure generator callback. User can overwrite this method to add more callbacks."""
        return self._auto_generator_callbacks(save_dir)

    def run_one_task_sample(self, sample: Any) -> Any:
        r"""Run one training sample. Used for debugging and testing."""
        training = self.task.training
        # test training
        self.task.train()
        task_call, kwargs = self.handle_one_task_sample(sample)
        output = task_call(**kwargs)
        if not isinstance(output, Parameter):
            warnings.warn(f"Output is not a Parameter in training mode: {output}")
        # eval mode
        self.task.eval()
        task_call, kwargs = self.handle_one_task_sample(sample)
        output = task_call(**kwargs)
        if isinstance(output, Parameter):
            warnings.warn(f"Output is a Parameter in evaluation mode: {output}")
        # reset training
        self.task.train(training)

    def run_one_loss_sample(self, sample: Any, y_pred: Any) -> Any:
        r"""Run one loss sample. Used for debugging and testing."""
        loss_call, kwargs = self.handle_one_loss_sample(sample, y_pred)
        return loss_call(**kwargs)

    def _find_all_generators(self) -> List[Tuple[str, "Generator"]]:
        r"""Find all generators automatically from the task."""
        from adalflow.core import Generator

        all_generators: List[Tuple[str, Generator]] = []
        for name, comp in self.task.named_components():
            if isinstance(comp, Generator):
                all_generators.append((name, comp))
        return all_generators

    def _auto_generator_callbacks(self, save_dir: str = "traces"):
        r"""Automatically generate callbacks."""
        from adalflow.core.types import GeneratorOutput
        from adalflow.tracing.generator_call_logger import (
            GeneratorCallLogger,
        )
        from functools import partial

        all_generators = self._find_all_generators()

        print(f"all_generators: {all_generators}")

        def _on_complete_callback(
            output: GeneratorOutput,
            input: Dict[str, Any],
            prompt_kwargs: Dict[str, Any],
            model_kwargs: Dict[str, Any],
            logger_call: Callable,
        ):
            r"""Log the generator output."""
            logger_call(
                output=output,
                input=input,
                prompt_kwargs=prompt_kwargs,
                model_kwargs=model_kwargs,
            )

        # Register the callback for each generator
        call_logger = GeneratorCallLogger(save_dir=save_dir)
        file_paths = []
        for name, generator in all_generators:
            call_logger.register_generator(name, f"{name}_call")
            logger_call = partial(call_logger.log_call, name)
            generator.register_callback(
                "on_complete", partial(_on_complete_callback, logger_call=logger_call)
            )
            file_path = call_logger.get_log_location(name)
            file_paths.append(file_path)
            print(
                f"Registered callback for {name}, file path: {file_path}",
                end="\n",
            )
        return file_paths

    def configure_demo_optimizer_helper(self) -> List[DemoOptimizer]:
        from adalflow.optim.few_shot.bootstrap_optimizer import BootstrapFewShot
        from adalflow.optim.parameter import ParameterType

        parameters = []
        for name, param in self.task.named_parameters():
            param.name = name
            if not param.param_type == ParameterType.DEMOS:
                continue
            parameters.append(param)
        if not parameters:
            raise ValueError("No demo parameters found.")
        do = BootstrapFewShot(params=parameters)
        return [do]

    def configure_text_optimizer_helper(
        self, model_client: "ModelClient", model_kwargs: Dict[str, Any]
    ) -> List[TextOptimizer]:
        from adalflow.optim.text_grad.tgd_optimer import TGDOptimizer
        from adalflow.optim.parameter import ParameterType

        parameters = []
        for name, param in self.task.named_parameters():
            param.name = name
            if not param.param_type == ParameterType.PROMPT:
                continue
            parameters.append(param)
        if not parameters:
            raise ValueError(
                "No text parameters found. Please define a demo parameter for your generator."
            )
        to = TGDOptimizer(
            params=parameters, model_client=model_client, model_kwargs=model_kwargs
        )
        return [to]
