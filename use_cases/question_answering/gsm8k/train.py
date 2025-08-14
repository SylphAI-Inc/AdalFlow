from typing import Any, Callable, Dict, Optional, Tuple
import adalflow as adal
from adalflow.datasets.types import GSM8KData as Example
from adalflow.datasets.gsm8k import GSM8K
from adalflow.eval.answer_match_acc import AnswerMatchAcc


def load_datasets():
    train_data = GSM8K(split="train", size=100)
    val_data = GSM8K(split="val", size=50)
    test_data = GSM8K(split="test", size=100)
    return train_data, val_data, test_data


class StandardTrain(adal.AdalComponent):
    __doc__ = """The standard training component can be used by many tasks that has a standard Data class and a exact match evaluation function."""

    def __init__(
        self,
        task: adal.Component,
        backward_engine_model_config: Optional[Dict] = None,
        teacher_model_config: Optional[Dict] = None,
        text_optimizer_model_config: Optional[Dict] = None,
        config: Optional[Dict] = None,
    ):
        task = GSM8KTask(**config)
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="exact_match: 1 if str(y) == str(y_gt) else 0",
        )
        super().__init__(task=task, eval_fn=eval_fn, loss_fn=loss_fn)

        self.backward_engine_model_config = backward_engine_model_config
        self.teacher_model_config = teacher_model_config
        self.text_optimizer_model_config = text_optimizer_model_config

    def prepare_task(self, sample: Example) -> Tuple[Callable, Dict[str, Any]]:
        return self.task.bicall, {"question": sample.question, "id": sample.id}

    def prepare_eval(
        self, sample: Example, y_pred: adal.GeneratorOutput
    ) -> Tuple[float, Dict[str, Any]]:
        y_label = ""
        if y_pred is not None and y_pred.data is not None:
            y_label = y_pred.data
        # printc(f"y_label: {y_label}, y_gt: {sample.answer}")
        return self.eval_fn, {"y": y_label, "y_gt": sample.answer}

    def prepare_loss(
        self, sample: Example, pred: adal.Parameter
    ) -> Tuple[Callable, Dict[str, Any]]:
        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.answer,
            eval_input=sample.answer,
            requires_opt=False,
        )
        pred.eval_input = pred.data.data
        return self.loss_fn, {"kwargs": {"y": pred, "y_gt": y_gt}, "id": sample.id}


from use_cases.question_answering.gsm8k.task import GSM8KTask

from use_cases.config import gpt_3_model, gpt_o3_mini_model


def train():
    train_data, val_data, test_data = load_datasets()

    # Create Anthropic client configuration instead of GPT-3
    anthropic_config = {
        "model_client": adal.AnthropicAPIClient(),
        "model_kwargs": {
            "model": "claude-3-5-sonnet-20241022",
            # "max_tokens": 2000,
            "temperature": 0.0,
        }
    }

    task = GSM8KTask(**anthropic_config)
    adal_component = StandardTrain(
        task=task,
        backward_engine_model_config=anthropic_config,
        text_optimizer_model_config=anthropic_config,
        config=anthropic_config,
    )
    trainer = adal.Trainer(
        adaltask=adal_component,
        strategy="random",
        # max_steps=10,
        max_steps=1, 
        text_optimizers_config_kwargs={"max_past_history": 5},
    )
    trainer.fit(
        train_dataset=train_data,
        val_dataset=val_data,
        test_dataset=test_data,
        debug=False,
        # resume_from_ckpt=None,
        # resume_from_ckpt="/Users/jinhakim/.adalflow/ckpt/StandardTrain/random_max_steps_1_e35e9_run_1.json",
    )


if __name__ == "__main__":
    train()
