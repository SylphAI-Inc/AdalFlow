from typing import Any, Callable, Dict, Tuple
import adalflow as adal
from use_cases.classification.trec_task_structured_output import (
    TRECClassifierStructuredOutput,
)

from use_cases.classification.data import load_datasets, TRECExtendedData

from adalflow.eval.answer_match_acc import AnswerMatchAcc
from LightRAG.use_cases.config import (
    gpt_3_model,
    gpt_4o_model,
)


class TrecClassifierAdal(adal.AdalComponent):
    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        backward_engine_model_config: Dict,
        text_optimizer_model_config: Dict,
    ):
        task = TRECClassifierStructuredOutput(model_client, model_kwargs)
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="exact_match: 1 if str(y) == str(y_gt) else 0",
        )
        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            backward_engine_model_config=backward_engine_model_config,
            text_optimizer_model_config=text_optimizer_model_config,
        )

    def handle_one_task_sample(self, sample: TRECExtendedData):
        return self.task.call, {"question": sample.question, "id": sample.id}

    def evaluate_one_sample(
        self, sample: TRECExtendedData, y_pred: adal.GeneratorOutput
    ) -> float:
        y_label = -1
        if y_pred and y_pred.data is not None and y_pred.data.class_index is not None:
            y_label = y_pred.data.class_index
        return self.eval_fn(y_label, sample.class_index)

    def handle_one_loss_sample(
        self, sample: Any, y_pred: adal.Parameter, *args, **kwargs
    ) -> Tuple[Callable[..., Any], Dict]:
        # prepare for evaluation
        full_response = y_pred.full_response
        y_label = -1
        if (
            full_response
            and full_response.data is not None
            and full_response.data.class_index is not None
        ):
            y_label = int(full_response.data.class_index)

        y_pred.eval_input = y_label
        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.class_index,
            eval_input=sample.class_index,
            requires_opt=False,
        )
        return self.loss_fn, {"kwargs": {"y": y_pred, "y_gt": y_gt}}

    def configure_backward_engine(self):
        super().configure_backward_engine_helper(**self.backward_engine_model_config)

    def configure_optimizers(self):
        to = super().configure_text_optimizer_helper(**self.text_optimizer_model_config)
        return to


def train(
    model_client: adal.ModelClient,
    model_kwargs: Dict,
    train_batch_size=4,  # larger batch size is not that effective, probably because of llm's lost in the middle
    raw_shots: int = 0,
    bootstrap_shots: int = 1,
    max_steps=1,
    num_workers=4,
    strategy="random",
    debug=False,
):
    adal_component = TrecClassifierAdal(
        model_client=model_client,
        model_kwargs=model_kwargs,
        text_optimizer_model_config=gpt_4o_model,
        backward_engine_model_config=gpt_4o_model,
    )
    print(adal_component)
    trainer = adal.Trainer(
        train_batch_size=train_batch_size,
        adaltask=adal_component,
        strategy=strategy,
        max_steps=max_steps,
        num_workers=num_workers,
        raw_shots=raw_shots,
        bootstrap_shots=bootstrap_shots,
        debug=debug,
        weighted_sampling=True,
    )
    print(trainer)

    train_dataset, val_dataset, test_dataset = load_datasets()
    trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        debug=debug,
    )


if __name__ == "__main__":
    train(**gpt_3_model, debug=False, max_steps=8, strategy="constrained")
