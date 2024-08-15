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
from adalflow.core.types import GeneratorOutput
from adalflow.optim.optimizer import Optimizer
from adalflow.optim.loss_component import LossComponent
from adalflow.optim.types import PromptData
from adalflow.eval.base import EvaluationResult

from adalflow.optim.optimizer import DemoOptimizer, TextOptimizer


class AdalComponent(Component):
    """Define a train, eval, and test step for a task pipeline.

    This serves the following purposes:
    1. Organize all parts for training a task pipeline organized in one place.
    2. Help with debugging and testing before the actual training.
    """

    task: Component
    # evaluator: Optional[BaseEvaluator]
    eval_fn: Optional[Callable]
    loss_fn: Optional[LossComponent]
    backward_engine: Optional["BackwardEngine"]
    _demo_optimizers: Optional[List[DemoOptimizer]]
    _text_optimizers: Optional[List[TextOptimizer]]

    def __init__(
        self,
        task: Component,
        # evaluator: Optional[BaseEvaluator] = None,
        eval_fn: Optional[Callable] = None,
        loss_fn: Optional[LossComponent] = None,
        backward_engine: Optional["BackwardEngine"] = None,
        backward_engine_model_config: Optional[Dict] = None,
        teacher_model_config: Optional[Dict] = None,
        text_optimizer_model_config: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.task = task
        # self.evaluator = evaluator
        self.eval_fn = eval_fn
        self.loss_fn = loss_fn
        self.backward_engine = backward_engine
        if backward_engine and not isinstance(backward_engine, "BackwardEngine"):
            raise ValueError(
                f"backward_engine is not a BackwardEngine: {backward_engine}"
            )
        self.backward_engine_model_config = backward_engine_model_config
        self.teacher_model_config = teacher_model_config
        self.text_optimizer_model_config = text_optimizer_model_config

    def _get_param_values(self) -> List[PromptData]:
        r"""Get the current values of the parameters."""
        return [
            PromptData(p.id, p.name, p.data, p.requires_opt)
            for p in self.task.parameters()
            # if p.requires_opt
        ]

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

        Need to ensure y_pred is a Parameter, and the real input to use
        for y_gt and y_pred is `eval_input`.
        Make sure it is setup.

        Example:

        .. code-block:: python

            # "y" and "y_gt" are arguments needed
            #by the eval_fn inside of the loss_fn if it is a EvalFnToTextLoss

            def handle_one_loss_sample(self, sample: Example, pred: adal.Parameter) -> Dict:
                # prepare gt parameter
                y_gt = adal.Parameter(
                    name="y_gt",
                    data=sample.answer,
                    eval_input=sample.answer,
                    requires_opt=False,
                )

                # pred's full_response is the output of the task pipeline which is GeneratorOutput
                pred.eval_input = pred.full_response.data
                return self.loss_fn, {"kwargs": {"y": y_gt, "y_pred": pred}}
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

    def evaluate_samples(
        self, samples: Any, y_preds: List, metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        r"""Run evaluation on samples. Use ``evaluate_one_sample`` defined by the user.

        Metadata is used for storing context that you can find from generator input.

        Note:
            ensure it supports both Tuple(batch) and a list of any type (fits for datasets).
        """
        if metadata is None:
            acc_list = [
                self.evaluate_one_sample(sample, y_pred)
                for sample, y_pred in zip(samples, y_preds)
            ]
        else:
            acc_list = [
                self.evaluate_one_sample(sample, y_pred, metadata=metadata)
                for sample, y_pred in zip(samples, y_preds)
            ]
        avg_score = float(np.mean(np.array(acc_list)))
        return EvaluationResult(avg_score=avg_score, per_item_scores=acc_list)

    def _train_step(
        self,
        batch,
        batch_idx,
        num_workers: int = 2,
    ):
        r"""Applies to both train and eval mode.

        If you require self.task.train() to be called before training, you can override this method as:

        .. code-block:: python

            def train_step(self, batch, batch_idx, num_workers: int = 2) -> List:
                self.task.train()
                return super().train_step(batch, batch_idx, num_workers)
        """
        from adalflow.optim.parameter import Parameter

        self.task.train()

        y_preds = [None] * len(batch)
        samples = [None] * len(batch)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            tqdm_loader = tqdm(batch, total=len(batch), desc="Loading Data")
            for i, sample in enumerate(tqdm_loader):
                task_call, kwargs = self.handle_one_task_sample(sample)
                future = executor.submit(task_call, **kwargs)
                futures.append((future, i, sample))  # preserve the order of the samples

            tqdm_loader = tqdm(
                total=len(futures),
                position=0,
                desc="Training",
            )
            for future, i, sample in futures:
                y_pred = future.result()
                y_preds[i] = y_pred  # Place the prediction in the correct position
                samples[i] = sample  # Keep the sample order aligned
                if not isinstance(y_pred, Parameter):
                    raise ValueError(f"y_pred_{i} is not a Parameter, {y_pred}")

                if hasattr(y_pred, "full_response") and isinstance(
                    y_pred.full_response, GeneratorOutput
                ):
                    if y_pred.full_response.id is not None:
                        y_pred_sample_id = y_pred.full_response.id
                        assert (
                            y_pred_sample_id == sample.id
                        ), f"ID mismatch: {y_pred_sample_id} != {sample.id}, type: {type(y_pred)}"

                tqdm_loader.update(1)  # Update the progress bar

        return y_preds

    def pred_step(
        self,
        batch,
        batch_idx,
        num_workers: int = 2,
        running_eval: bool = False,
        min_score: Optional[float] = None,
    ):
        r"""Applies to both train and eval mode.

        If you require self.task.train() to be called before training, you can override this method as:

        .. code-block:: python

            def train_step(self, batch, batch_idx, num_workers: int = 2) -> List:
                self.task.train()
                return super().train_step(batch, batch_idx, num_workers)
        """
        from adalflow.optim.parameter import Parameter

        self.task.eval()

        y_preds = [None] * len(batch)
        samples = [None] * len(batch)
        completed_indices = set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            tqdm_loader = tqdm(batch, total=len(batch), desc="Loading Data")
            for i, sample in enumerate(tqdm_loader):
                task_call, kwargs = self.handle_one_task_sample(sample)
                future = executor.submit(task_call, **kwargs)
                futures.append((future, i, sample))  # preserve the order of the samples

            tqdm_loader = tqdm(
                total=len(futures),
                position=0,
                desc=f"Evaluating step: {batch_idx}",
            )
            for future, i, sample in futures:
                y_pred = future.result()
                y_preds[i] = y_pred  # Place the prediction in the correct position
                samples[i] = sample  # Keep the sample order aligned
                # check the ordering

                assert (
                    y_pred.id == sample.id
                ), f"ID mismatch: {y_pred.id} != {sample.id}, type: {type(y_pred)}"
                completed_indices.add(i)  # Mark this index as completed

                if running_eval and not isinstance(y_pred, Parameter):
                    # Perform running evaluation only on completed samples
                    sorted_indices = sorted(completed_indices)
                    completed_y_preds = [y_preds[idx] for idx in sorted_indices]
                    completed_samples = [samples[idx] for idx in sorted_indices]
                    # for y_pred, sample in zip(completed_y_preds, completed_samples):
                    #     print(f"y_pred: {y_pred.data}, sample: {sample.answer}")
                    # for y_pred, sample in zip(completed_y_preds, completed_samples):
                    #     if y_pred.id != sample.id:
                    #         raise ValueError(
                    #             f"ID mismatch: {y_pred.id} != {sample.id}, type: {type(y_pred)}"
                    #         )
                    #     print(f"y_pred: {y_pred.data}, sample: {sample.answer}")
                    # TODO: each time only call evaluate on one sample
                    eval_score = self.evaluate_samples(
                        completed_samples, completed_y_preds
                    ).avg_score

                    remaining_samples = len(batch) - len(completed_indices)
                    max_score = (
                        eval_score * len(completed_indices) + remaining_samples
                    ) / len(batch)

                    if min_score is not None and max_score < min_score:
                        break

                    tqdm_loader.set_description(
                        f"Evaluating step({batch_idx}): {round(eval_score,4)} across {len(completed_indices)} samples, Max potential: {round(max_score,4)}"
                    )
                else:
                    tqdm_loader.set_description(f"Evaluating step({batch_idx})")

                tqdm_loader.update(1)  # Update the progress bar

        sorted_indices = sorted(completed_indices)
        completed_y_preds = [y_preds[idx] for idx in sorted_indices]
        completed_samples = [samples[idx] for idx in sorted_indices]

        return completed_y_preds, completed_samples

    def train_step(self, batch, batch_idx, num_workers: int = 2) -> List:
        self.task.train()
        y_preds = self._train_step(batch, batch_idx, num_workers)
        for i, y_pred in enumerate(y_preds):
            try:
                y_pred.name += f"y_pred_{i}"
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
        completed_y_preds, completed_samples = self.pred_step(
            batch, batch_idx, num_workers, running_eval=True, min_score=minimum_score
        )
        eval_results = self.evaluate_samples(
            samples=completed_samples, y_preds=completed_y_preds
        )
        return eval_results

    def loss_step(
        self, batch, y_preds: List["Parameter"], batch_idx, num_workers: int = 2
    ) -> List["Parameter"]:
        r"""Calculate the loss for the batch."""
        from adalflow.optim.parameter import Parameter

        losses = [None] * len(batch)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            tqdm_loader = tqdm(
                zip(batch, y_preds), total=len(batch), desc="Loading Data"
            )
            for i, (sample, y_pred) in enumerate(tqdm_loader):
                loss_forward, kwargs = self.handle_one_loss_sample(sample, y_pred)
                future = executor.submit(loss_forward, **kwargs)
                futures.append((future, i, sample))
            tqdm_loader = tqdm(
                total=len(futures),
                position=0,
                desc="Calculating Loss",
            )
            for future, i, sample in futures:

                loss = future.result()
                if not isinstance(loss, Parameter):
                    raise ValueError(f"Loss is not a Parameter: {loss}")
                losses[i] = loss
                tqdm_loader.update(1)
        return losses

    def configure_teacher_generator(self):
        r"""Configure a teach generator for all generators in the task for bootstrapping examples.

        You can call `configure_teacher_generator_helper` to easily configure it by passing the model_client and model_kwargs.
        """
        raise NotImplementedError(
            "configure_teacher_generator method is not implemented"
        )

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

    def configure_callbacks(self, save_dir: Optional[str] = "traces", *args, **kwargs):
        """In default we config the failure generator callback. User can overwrite this method to add more callbacks."""
        from adalflow.utils.global_config import get_adalflow_default_root_path
        import os

        if not save_dir:
            save_dir = "traces"
            save_dir = os.path.join(get_adalflow_default_root_path(), save_dir)
        print(f"Saving traces to {save_dir}")
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

        def _on_completion_callback(
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

        file_paths = []
        for name, generator in all_generators:
            call_logger = GeneratorCallLogger(save_dir=save_dir)
            call_logger.reset()
            call_logger.register_generator(name)
            logger_call = partial(call_logger.log_call, name)
            generator.register_callback(
                "on_complete", partial(_on_completion_callback, logger_call=logger_call)
            )
            file_path = call_logger.get_log_location(name)
            file_paths.append(file_path)
            print(
                f"Registered callback for {name}, file path: {file_path}",
                end="\n",
            )
        return file_paths

    def configure_demo_optimizer_helper(self) -> List[DemoOptimizer]:
        r"""One demo optimizer can handle multiple demo parameters.
        But the demo optimizer will only have one dataset (trainset) configured by the Trainer.

        If users want to use different trainset for different demo optimizer,
        they can configure it by themselves.
        """
        from adalflow.optim.few_shot.bootstrap_optimizer import BootstrapFewShot
        from adalflow.optim.parameter import ParameterType

        parameters = []
        for name, param in self.task.named_parameters():
            param.name = name
            if not param.param_type == ParameterType.DEMOS:
                continue
            parameters.append(param)
        if len(parameters) == 0:
            print("No demo parameters found.")
            return []
        do = BootstrapFewShot(params=parameters)
        return [do]

    def configure_text_optimizer_helper(
        self, model_client: "ModelClient", model_kwargs: Dict[str, Any]
    ) -> List[TextOptimizer]:
        r"""One text optimizer can handle multiple text parameters."""
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

    def _extra_repr(self):
        s = f"eval_fn: {self.eval_fn.__name__}, backward_engine: {self.backward_engine}, "
        s += f"backward_engine_model_config: {self.backward_engine_model_config}, teacher_model_config: {self.teacher_model_config}, text_optimizer_model_config: {self.text_optimizer_model_config}"
        return s
