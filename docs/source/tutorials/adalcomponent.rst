AdalComponent
================
``AdalComponent`` is where users put together the task pipeline to optimize, the train, validate steps along with optimizers, evaluator, loss function, and backward engine.
It is inspired by `PyTorch Lightning`'s `LightningModule`. It enables generalized and smooth integration to the ``Trainer``.


Here is one example:

.. code-block:: python

    class HotPotQARAGAdal(AdalComponent):
        # TODO: move teacher model or config in the base class so users dont feel customize too much
        def __init__(self, task: Component, teacher_model_config: dict):
            super().__init__()
            self.task = task
            self.teacher_model_config = teacher_model_config

            self.evaluator = AnswerMatchAcc("fuzzy_match")
            self.eval_fn = self.evaluator.compute_single_item

        def handle_one_task_sample(
            self, sample: HotPotQAData
        ) -> Any:  # TODO: auto id, with index in call train examples
            return self.task, {"question": sample.question, "id": sample.id}

        def handle_one_loss_sample(
            self, sample: HotPotQAData, y_pred: Any
        ) -> Tuple[Callable, Dict]:
            return self.loss_fn.forward, {
                "kwargs": {
                    "y": y_pred,
                    "y_gt": Parameter(
                        data=sample.answer,
                        role_desc="The ground truth(reference correct answer)",
                        alias="y_gt",
                        requires_opt=False,
                    ),
                }
            }

        def configure_optimizers(self, *args, **kwargs):

            # TODO: simplify this, make it accept generator
            parameters = []
            for name, param in self.task.named_parameters():
                param.name = name
                parameters.append(param)
            do = BootstrapFewShot(params=parameters)
            return [do]

        def evaluate_one_sample(
            self, sample: Any, y_pred: Any, metadata: Dict[str, Any]
        ) -> Any:

            # we need "context" be passed as metadata
            # print(f"sample: {sample}, y_pred: {y_pred}")
            # convert pred to Dspy structure

            # y_obj = convert_y_pred_to_dataclass(y_pred)
            # print(f"y_obj: {y_obj}")
            # raise ValueError("Stop here")
            if metadata:
                return self.eval_fn(sample, y_pred, metadata)
            return self.eval_fn(sample, y_pred)

        def configure_teacher_generator(self):
            super().configure_teacher_generator(**self.teacher_model_config)

        def configure_loss_fn(self):
            self.loss_fn = EvalFnToTextLoss(
                eval_fn=self.eval_fn,
                eval_fn_desc="ObjectCountingEvalFn, Output accuracy score: 1 for correct, 0 for incorrect",
                backward_engine=None,
            )
