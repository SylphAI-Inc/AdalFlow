"""AdalComponent provides an interface to compose different parts, from eval_fn, train_step, loss_step, optimizers, backward engine, teacher generator, etc to work with Trainer."""

from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
import concurrent
from tqdm import tqdm
import numpy as np
import warnings
import logging as log

if TYPE_CHECKING:
    from adalflow.core.model_client import ModelClient
    from adalflow.core.generator import Generator, BackwardEngine, BackwardPassSetup
    from adalflow.optim.parameter import Parameter

from adalflow.core.component import Component
from adalflow.core.types import GeneratorOutput
from adalflow.optim.optimizer import Optimizer
from adalflow.optim.loss_component import LossComponent
from adalflow.optim.types import PromptData
from adalflow.eval.base import EvaluationResult
from adalflow.optim.grad_component import GradComponent
from adalflow.utils import printc

from adalflow.optim.optimizer import DemoOptimizer, TextOptimizer


log = log.getLogger(__name__)


class AdalComponent(Component):
    __doc__ = """Define a train, eval, and test step for a task pipeline.

    This serves the following purposes:
    1. Organize all parts for training a task pipeline in one place.
    2. Help with debugging and testing before the actual training.
    3. Adds multi-threading support for training and evaluation.

    It has no need on call, forward, bicall, or __call__, so we need to overwrite the base ones.
    """

    task: Component
    # evaluator: Optional[BaseEvaluator]
    eval_fn: Optional[Callable]  # final eval score
    loss_fn: Optional[LossComponent]
    loss_eval_fn: Optional[Callable]  # loss eval score and training subset eval fn
    backward_engine: Optional["BackwardEngine"]
    _demo_optimizers: Optional[List[DemoOptimizer]]
    _text_optimizers: Optional[List[TextOptimizer]]

    def __init__(
        self,
        task: Component,
        # evaluator: Optional[BaseEvaluator] = None,
        eval_fn: Optional[Callable] = None,
        loss_eval_fn: Optional[Callable] = None,
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
        self.loss_eval_fn = loss_eval_fn
        self.loss_fn = loss_fn
        self.backward_engine = backward_engine
        if backward_engine and not isinstance(backward_engine, "BackwardEngine"):
            raise ValueError(
                f"backward_engine is not a BackwardEngine: {backward_engine}"
            )
        self.backward_engine_model_config = backward_engine_model_config
        self.teacher_model_config = teacher_model_config
        self.text_optimizer_model_config = text_optimizer_model_config
        self._demo_optimizers = None
        self._text_optimizers = None

    def _set_param_values(self, prompts: List[PromptData]):
        r"""Set the parameters for the task. Used to resume from ckpt."""

        params_dict = {p.name: p for p in prompts}

        for name, param in self.task.named_parameters():
            if name in params_dict:
                param.update_value(params_dict[name].data)

    def _get_param_values(self) -> List[PromptData]:
        r"""Get the current values of the parameters."""
        return [
            PromptData(p.id, p.name, p.data, p.requires_opt)
            for p in self.task.parameters()
            if p.requires_opt
        ]

    def prepare_task(self, sample: Any, *args, **kwargs) -> Tuple[Callable, Dict]:
        r"""Tell Trainer how to call the task in both training and inference mode.

        Return a task call and kwargs for one training sample.

        If you just need to eval, ensure the Callable has the inference mode.
        If you need to also train, ensure the Callable has the training mode which returns a Parameter and mainly
        call forward for all subcomponents within the task.

        Example:

        .. code-block:: python

            def prepare_task(self, sample: Any, *args, **kwargs) -> Tuple[Callable, Dict]:
                return self.task, {"x": sample.x}
        """

        raise NotImplementedError("prepare_task method is not implemented")

    def prepare_loss(
        self, sample: Any, y_pred: "Parameter", *args, **kwargs
    ) -> Tuple[Callable, Dict]:
        r"""Tell Trainer how to calculate the loss in the training mode.

        Return a loss call and kwargs for one loss sample.

        Need to ensure y_pred is a Parameter, and the real input to use
        for y_gt and y_pred is `eval_input`.
        Make sure it is setup.

        Example:

        .. code-block:: python

            # "y" and "y_gt" are arguments needed
            #by the eval_fn inside of the loss_fn if it is a EvalFnToTextLoss

            def prepare_loss(self, sample: Example, pred: adal.Parameter) -> Dict:
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
        raise NotImplementedError("prepare_loss method is not implemented")

    # TODO: Support multiple eval_fn with different metrics. using a dict[str, (Callable, Dict)] to store them.
    def prepare_eval(self, sample: Any, y_pred: Any, *args, **kwargs) -> float:
        r"""Tell Trainer how to eval in inference mode.
        Return the eval_fn and kwargs for one evaluation sample.

        Ensure the eval_fn is a callable that takes the predicted output and the ground truth output.
        Ensure the kwargs are setup correctly.

        """
        raise NotImplementedError("prepare_eval method is not implemented")

    def prepare_loss_eval(self, sample: Any, y_pred: Any, *args, **kwargs) -> float:
        r"""Tell Trainer how to eval in inference mode.
        Return the eval_fn and kwargs for one evaluation sample.

        Ensure the eval_fn is a callable that takes the predicted output and the ground truth output.
        Ensure the kwargs are setup correctly.

        """
        raise NotImplementedError("prepare_loss_eval method is not implemented")

    # def configure_optimizers(self, *args, **kwargs) -> Optimizer:
    #     r"""Note: When you use text optimizor, ensure you call `configure_backward_engine_engine` too."""
    #     raise NotImplementedError("configure_optimizers method is not implemented")

    def configure_optimizers(self, *args, **text_optimizer_kwargs) -> List[Optimizer]:
        r"""Note: When you use text optimizor, ensure you call `configure_backward_engine_engine` too."""
        if self._demo_optimizers is None:
            self._demo_optimizers = self.configure_demo_optimizer_helper()
        if self._text_optimizers is None:
            if not self.text_optimizer_model_config:
                raise ValueError("Text optimizer model config is not configured.")
            if not self.text_optimizer_model_config.get("model_client"):
                raise ValueError("Model client is not configured.")
            if not self.text_optimizer_model_config.get("model_kwargs"):
                raise ValueError("Model kwargs is not configured.")

            self._text_optimizers = self.configure_text_optimizer_helper(
                **self.text_optimizer_model_config, **text_optimizer_kwargs
            )
        return self._demo_optimizers + self._text_optimizers

    def configure_backward_engine(self, *args, **kwargs):
        r"""Configure a backward engine for all GradComponent in the task for bootstrapping examples."""
        # check if backward engine is already configured
        if self.backward_engine:
            log.warning("Backward engine is already configured.")
        if not self.backward_engine_model_config:
            raise ValueError("Backward engine model config is not configured.")
        if not self.backward_engine_model_config.get("model_client"):
            raise ValueError("Model client is not configured.")
        if not self.backward_engine_model_config.get("model_kwargs"):
            raise ValueError("Model kwargs is not configured.")
        self.configure_backward_engine_helper(
            model_client=self.backward_engine_model_config["model_client"],
            model_kwargs=self.backward_engine_model_config["model_kwargs"],
            backward_pass_setup=kwargs.get("backward_pass_setup", None),
        )

    def disable_backward_engine(self):
        r"""Disable the backward engine for all GradComponent in the task.
        No more gradients generation."""
        self.disable_backward_engine_helper()

    # def configure_backward_engine(self, *args, **kwargs):
    #     raise NotImplementedError("configure_backward_engine method is not implemented")

    def evaluate_samples(
        self,
        samples: Any,
        y_preds: List,
        metadata: Optional[Dict[str, Any]] = None,
        num_workers: int = 2,
        use_loss_eval_fn: bool = False,
    ) -> EvaluationResult:
        r"""Evaluate predictions against the ground truth samples.
        Run evaluation on samples using parallel processing. Utilizes ``prepare_eval`` defined by the user.

        Metadata is used for storing context that you can find from generator input.

        Args:
            samples (Any): The input samples to evaluate.
            y_preds (List): The predicted outputs corresponding to each sample.
            metadata (Optional[Dict[str, Any]]): Optional metadata dictionary.
            num_workers (int): Number of worker threads for parallel processing.

        Returns:
            EvaluationResult: An object containing the average score and per-item scores.
        """
        if use_loss_eval_fn and not self.loss_eval_fn:
            raise ValueError("Loss eval function is not configured.")

        from adalflow.optim.parameter import Parameter

        if not isinstance(y_preds, list) or len(y_preds) == 0:
            raise ValueError(f"y_preds is not a list or empty: {y_preds}")

        y_pred_0 = y_preds[0]
        if isinstance(y_pred_0, Parameter):
            raise ValueError(f"y_pred_0 should not be a Parameter: {y_pred_0}")

        acc_list = [None] * len(samples)  # Initialize accuracy list to hold results

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:

            # 1. submit all the tasks
            futures = {}

            for i, (sample, y_pred) in enumerate(zip(samples, y_preds)):

                if metadata is None:
                    if not use_loss_eval_fn:
                        eval_fn, kwargs = self.prepare_eval(sample, y_pred)
                    else:
                        eval_fn, kwargs = self.prepare_loss_eval(sample, y_pred)
                    future = executor.submit(eval_fn, **kwargs)
                    # future = executor.submit(self.evaluate_one_sample, sample, y_pred)
                else:
                    if not use_loss_eval_fn:
                        eval_fn, kwargs = self.prepare_eval(
                            sample, y_pred, metadata=metadata
                        )
                    else:

                        eval_fn, kwargs = self.prepare_eval(
                            sample, y_pred, metadata=metadata
                        )
                    future = executor.submit(eval_fn, **kwargs)
                    # future = executor.submit(
                    #     self.evaluate_one_sample, sample, y_pred, metadata=metadata
                    # )
                futures[future] = i

            # 2. collect the results, update the progress bar
            # Initialize progress bar once outside the loop
            progress_bar = tqdm(
                total=len(samples), desc="Evaluating", position=0, leave=True
            )

            for future in concurrent.futures.as_completed(futures):
                try:
                    i = futures[future]
                    acc_list[i] = (
                        future.result()
                    )  # Place the result in the correct position
                    progress_bar.update(
                        1
                    )  # Update progress bar after each result is collected
                except Exception as e:

                    progress_bar.close()
                    raise ValueError(f"Exception in task {i}: {e}")

        avg_score = float(np.mean(np.array(acc_list)))
        return EvaluationResult(avg_score=avg_score, per_item_scores=acc_list)

    # def evaluate_samples(
    #     self, samples: Any, y_preds: List, metadata: Optional[Dict[str, Any]] = None
    # ) -> EvaluationResult:
    #     r"""Run evaluation on samples. Use ``evaluate_one_sample`` defined by the user.

    #     Metadata is used for storing context that you can find from generator input.

    #     Note:
    #         ensure it supports both Tuple(batch) and a list of any type (fits for datasets).
    #     """
    #     from adalflow.optim.parameter import Parameter

    #     if not isinstance(y_preds, list) or len(y_preds) == 0:
    #         raise ValueError(f"y_preds is not a list or empty: {y_preds}")
    #     y_pred_0 = y_preds[0]
    #     if isinstance(y_pred_0, Parameter):
    #         raise ValueError(f"y_pred_0 should not be a Parameter: {y_pred_0}")
    #     if metadata is None:
    #         acc_list = [
    #             self.evaluate_one_sample(sample, y_pred)
    #             for sample, y_pred in zip(samples, y_preds)
    #         ]
    #     else:
    #         acc_list = [
    #             self.evaluate_one_sample(sample, y_pred, metadata=metadata)
    #             for sample, y_pred in zip(samples, y_preds)
    #         ]
    #     avg_score = float(np.mean(np.array(acc_list)))
    #     return EvaluationResult(avg_score=avg_score, per_item_scores=acc_list)

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
                task_call, kwargs = self.prepare_task(sample)
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
        use_loss_eval_fn: bool = False,
    ) -> Tuple[List["Parameter"], List, Dict[int, float]]:
        r"""Applies to only the eval mode.

        Args:
            batch (Any): The input batch to predict.
            batch_idx (int): The index of the batch.
            num_workers (int): Number of worker threads for parallel processing.
            running_eval: bool = False,
        Returns:
            Tuple[List["Parameter"], List, Dict[int, float]]: The predicted outputs, the samples, and the scores.
        """
        from adalflow.optim.parameter import Parameter

        self.task.eval()

        y_preds = [None] * len(batch)
        samples = [None] * len(batch)
        completed_indices = set()
        index_to_score = {}  # for running evaluation

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 1. submit all the tasks
            futures = []
            tqdm_loader = tqdm(batch, total=len(batch), desc="Loading Data")
            for i, sample in enumerate(tqdm_loader):
                task_call, kwargs = self.prepare_task(sample)
                future = executor.submit(task_call, **kwargs)
                futures.append((future, i, sample))  # preserve the order of the samples

            # 2. predict the results, update the progress bar
            tqdm_loader = tqdm(
                total=len(futures),
                position=0,
                desc=f"Prediting step: {batch_idx}",
            )
            for future, i, sample in futures:
                y_pred = future.result()
                y_preds[i] = y_pred  # Place the prediction in the correct position
                samples[i] = sample  # Keep the sample order aligned
                # check the ordering

                if isinstance(y_pred, Parameter):
                    raise ValueError(f"y_pred_{i} is a Parameter, {y_pred}")

                assert (
                    y_pred.id == sample.id
                ), f"ID mismatch: {y_pred.id} != {sample.id}, type: {type(y_pred)}"

                completed_indices.add(i)  # Mark this index as completed

                if running_eval and not isinstance(y_pred, Parameter):
                    # evaluate one sample

                    if not use_loss_eval_fn:
                        eval_fn, kwargs = self.prepare_eval(sample, y_pred)
                    else:
                        eval_fn, kwargs = self.prepare_loss_eval(sample, y_pred)
                    score = eval_fn(**kwargs)
                    index_to_score[i] = score
                    eval_score = np.mean(list(index_to_score.values())).item()

                    # for y_pred, sample in zip(completed_y_preds, completed_samples):
                    #     print(f"y_pred: {y_pred.data}, sample: {sample.answer}")
                    # for y_pred, sample in zip(completed_y_preds, completed_samples):
                    #     if y_pred.id != sample.id:
                    #         raise ValueError(
                    #             f"ID mismatch: {y_pred.id} != {sample.id}, type: {type(y_pred)}"
                    #         )
                    #     print(f"y_pred: {y_pred.data}, sample: {sample.answer}")

                    remaining_samples = len(batch) - len(completed_indices)
                    max_score = (
                        eval_score * len(completed_indices) + remaining_samples
                    ) / len(batch)

                    if min_score is not None and max_score < min_score:
                        break

                    tqdm_loader.set_description(
                        f"Predicting: step({batch_idx}): {round(eval_score,4)} across {len(completed_indices)} samples, Max potential: {round(max_score,4)}"
                    )
                else:
                    tqdm_loader.set_description(f"Predicting: step({batch_idx})")

                tqdm_loader.update(1)  # Update the progress bar

        sorted_indices = sorted(completed_indices)
        completed_y_preds = [y_preds[idx] for idx in sorted_indices]
        completed_samples = [samples[idx] for idx in sorted_indices]

        return completed_y_preds, completed_samples, index_to_score

    def train_step(self, batch, batch_idx, num_workers: int = 2) -> List:
        r"""Run a training step and return the predicted outputs.
        Likely a list of Parameters."""
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
        use_loss_eval_fn: bool = False,
    ) -> EvaluationResult:
        r"""
        Args:
            batch (Any): The input batch to validate, can be a whole dataset
            batch_idx (int): The index of the batch. or current_step
            num_workers (int): Number of worker threads for parallel processing.
            minimum_score (Optional[float]): The max potential score needs to be larger than this to continue evaluating.

        Evaluate a batch or the validate dataset by setting the batch=val_dataset.
        Uses self.eval_fn to evaluate the samples.
        If you require self.task.eval() to be called before validation, you can override this method as:

        .. code-block:: python

            def validation_step(self, batch, batch_idx, num_workers: int = 2) -> List:
                self.task.eval()
                return super().validation_step(batch, batch_idx, num_workers)
        """
        # TODO: let use decide which mode to be
        eval_fn = self.eval_fn
        if use_loss_eval_fn:
            eval_fn = self.loss_eval_fn
            if not eval_fn:
                raise ValueError("Loss eval function is not configured.")

        self.task.eval()
        self.task.use_teacher(mode=False)  # ensure the teacher is not used
        try:
            completed_y_preds, completed_samples, index_to_score = self.pred_step(
                batch,
                batch_idx,
                num_workers,
                running_eval=True,
                min_score=minimum_score,
                use_loss_eval_fn=use_loss_eval_fn,
            )
        except Exception as e:
            raise ValueError(f"Error in validation step: {e}")

        if index_to_score:
            # compute score from index_to_score

            avg_score = np.mean(list(index_to_score.values())).item()
            acc_list = [None] * len(index_to_score)
            for i, score in index_to_score.items():
                acc_list[i] = score
            acc_list = list(index_to_score.values())
            eval_results = EvaluationResult(
                avg_score=avg_score, per_item_scores=acc_list
            )
        else:
            try:

                eval_results = self.evaluate_samples(
                    samples=completed_samples,
                    y_preds=completed_y_preds,
                    num_workers=num_workers,
                    use_loss_eval_fn=use_loss_eval_fn,
                )
            except Exception as e:
                raise ValueError(f"Error in evaluation: {e}")
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
                loss_forward, kwargs = self.prepare_loss(sample, y_pred)
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

    # def configure_teacher_generator(self):
    #     r"""Configure a teach generator for all generators in the task for bootstrapping examples.

    #     You can call `configure_teacher_generator_helper` to easily configure it by passing the model_client and model_kwargs.
    #     """
    #     raise NotImplementedError(
    #         "configure_teacher_generator method is not implemented"
    #     )

    # use default implementation
    def configure_teacher_generator(self):
        r"""Configure a teach generator for all generators in the task for bootstrapping examples.

        You can call `configure_teacher_generator_helper` to easily configure it by passing the model_client and model_kwargs.
        """
        if not self.teacher_model_config:
            raise ValueError("Teacher model config is not configured.")
        if not self.teacher_model_config.get("model_client"):
            raise ValueError("Model client is not configured.")
        if not self.teacher_model_config.get("model_kwargs"):
            raise ValueError("Model kwargs is not configured.")
        print("Configuring teacher generator.")
        self.configure_teacher_generator_helper(**self.teacher_model_config)

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

    def disable_backward_engine_helper(self):
        r"""Disable the backward engine for all gradcomponents in the task."""
        all_grads = self._find_all_grad_components()
        for _, grad in all_grads:
            if hasattr(grad, "disable_backward_engine") and callable(
                getattr(grad, "disable_backward_engine", None)
            ):
                grad.disable_backward_engine()
        print("Backward engine disabled for GradComponents")

        if not self.loss_fn:
            raise ValueError("Loss function is not configured.")

        # configure it for loss_fn
        if self.loss_fn:
            self.loss_fn.disable_backward_engine()

    def configure_backward_engine_helper(
        self,
        model_client: "ModelClient",
        model_kwargs: Dict[str, Any],
        template: Optional[str] = None,
        backward_pass_setup: Optional["BackwardPassSetup"] = None,
    ):
        r"""Configure a backward engine for all generators in the task for bootstrapping examples."""
        from adalflow.core.generator import BackwardEngine

        self.backward_engine = BackwardEngine(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=template,
        )
        if backward_pass_setup is not None:
            self.backward_engine.update_default_backward_pass_setup(backward_pass_setup)

        # set all generator's backward engine

        all_grads = self._find_all_grad_components()
        for _, grad in all_grads:
            if hasattr(grad, "set_backward_engine") and callable(
                getattr(grad, "set_backward_engine", None)
            ):
                grad.set_backward_engine(self.backward_engine)
        print("Backward engine configured for GradComponents")

        if not self.loss_fn:
            raise ValueError("Loss function is not configured.")

        # configure it for loss_fn
        if self.loss_fn:
            self.loss_fn.set_backward_engine(self.backward_engine)

    def configure_callbacks(
        self, save_dir: Optional[str] = "traces", *args, **kwargs
    ) -> List[str]:
        """In default we config the failure generator callback. User can overwrite this method to add more callbacks."""
        from adalflow.utils.global_config import get_adalflow_default_root_path
        import os

        if not save_dir:
            save_dir = "traces"
            save_dir = os.path.join(get_adalflow_default_root_path(), save_dir)
        log.debug(f"Saving traces to {save_dir}")
        return self._auto_generator_callbacks(save_dir)

    def run_one_task_sample(self, sample: Any) -> Any:
        r"""Run one training sample. Used for debugging and testing."""
        training = self.task.training
        # test training
        self.task.train()
        task_call, kwargs = self.prepare_task(sample)
        output = task_call(**kwargs)
        if not isinstance(output, Parameter):
            warnings.warn(f"Output is not a Parameter in training mode: {output}")
        # eval mode
        self.task.eval()
        task_call, kwargs = self.prepare_task(sample)
        output = task_call(**kwargs)
        if isinstance(output, Parameter):
            warnings.warn(f"Output is a Parameter in evaluation mode: {output}")
        # reset training
        self.task.train(training)

    def run_one_loss_sample(self, sample: Any, y_pred: Any) -> Any:
        r"""Run one loss sample. Used for debugging and testing."""
        loss_call, kwargs = self.prepare_loss(sample, y_pred)
        return loss_call(**kwargs)

    def _find_all_generators(self) -> List[Tuple[str, "Generator"]]:
        r"""Find all generators automatically from the task."""
        from adalflow.core import Generator

        all_generators: List[Tuple[str, Generator]] = []
        for name, comp in self.task.named_components():
            if isinstance(comp, Generator):
                all_generators.append((name, comp))
        log.debug(f"all_generators: {all_generators}")
        return all_generators

    def _find_all_grad_components(self) -> List[Tuple[str, GradComponent]]:
        r"""Find all generators automatically from the task."""
        # from adalflow.core import Generator

        all_grads: List[Tuple[str, GradComponent]] = []
        for name, comp in self.task.named_components():
            if isinstance(comp, GradComponent) or isinstance(comp, GradComponent):
                all_grads.append((name, comp))
        log.debug(f"all_grads: {all_grads}")
        return all_grads

    def _auto_generator_callbacks(self, save_dir: str = "traces") -> List[str]:
        r"""Automatically generate callbacks."""
        from adalflow.core.types import GeneratorOutput
        from adalflow.tracing.generator_call_logger import (
            GeneratorCallLogger,
        )
        from functools import partial

        all_generators = self._find_all_generators()

        log.debug(f"all_generators: {all_generators}")

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
        call_logger = GeneratorCallLogger(save_dir=save_dir)
        for name, generator in all_generators:

            # call_logger.reset()
            call_logger.register_generator(name)
            logger_call = partial(call_logger.log_call, name)
            generator.register_callback(
                "on_complete", partial(_on_completion_callback, logger_call=logger_call)
            )
            file_path = call_logger.get_log_location(name)
            file_paths.append(file_path)
            log.debug(f"Registered callback for {name}, file path: {file_path}")
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
        self, model_client: "ModelClient", model_kwargs: Dict[str, Any], **kwargs
    ) -> List[TextOptimizer]:
        r"""Text optimizer hands prompt parameter type. One text optimizer can handle multiple text parameters."""
        from adalflow.optim.text_grad.tgd_optimizer import TGDOptimizer
        from adalflow.optim.parameter import ParameterType

        parameters = []
        for name, param in self.task.named_parameters():
            param.name = name
            if not param.param_type == ParameterType.PROMPT:
                continue
            parameters.append(param)
        if not parameters:
            return []

        to = TGDOptimizer(
            params=parameters,
            model_client=model_client,
            model_kwargs=model_kwargs,
            **kwargs,
        )

        printc(
            f"Text optimizer configured for {len(parameters)} parameters. names: { [(p.name, p.data) for p in parameters] }"
        )
        return [to]

    def _extra_repr(self):
        s = f"eval_fn: {self.eval_fn.__name__}, backward_engine: {self.backward_engine}, "
        s += f"backward_engine_model_config: {self.backward_engine_model_config}, teacher_model_config: {self.teacher_model_config}, text_optimizer_model_config: {self.text_optimizer_model_config}"
        return s

    def __call__(self, *args, **kwargs):
        pass

    def bicall(self, *args, **kwargs):
        pass

    def call(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass
