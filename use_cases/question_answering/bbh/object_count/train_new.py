from use_cases.question_answering.bbh.object_count.task import (
    ObjectCountTaskPipeline,
)

from use_cases.config import (
    gpt_3_model,
    gpt_4o_model,
)

from typing import Any, Callable, Dict, Optional, Tuple
import adalflow as adal

from adalflow.datasets.types import Example
from adalflow.eval.answer_match_acc import AnswerMatchAcc
from use_cases.question_answering.bbh.data import load_datasets


class ObjectCountAdalComponent(adal.AdalComponent):
    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        backward_engine_model_config: Optional[Dict] = None,
        teacher_model_config: Optional[Dict] = None,
        text_optimizer_model_config: Optional[Dict] = None,
    ):
        task = ObjectCountTaskPipeline(model_client, model_kwargs)
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
        return self.task.call, {"question": sample.question, "id": sample.id}

    def prepare_eval(
        self, sample: Example, y_pred: adal.GeneratorOutput
    ) -> Tuple[float, Dict[str, Any]]:
        y_label = -1
        if y_pred and y_pred.data:
            y_label = y_pred.data
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
        pred.eval_input = pred.full_response.data
        return self.loss_fn, {"kwargs": {"y": pred, "y_gt": y_gt}}


# TODO: make the train diagnose on the student model and the teacher model automatcally
# in the trainer
def train_diagnose(
    model_client: adal.ModelClient,
    model_kwargs: Dict,
) -> Dict:
    from use_cases.question_answering.bbh.data import load_datasets

    trainset, valset, testset = load_datasets()

    adal_component = ObjectCountAdalComponent(model_client, model_kwargs)
    trainer = adal.Trainer(adaltask=adal_component)
    trainer.diagnose(dataset=trainset, split="train")
    trainer.diagnose(dataset=valset, split="val")
    trainer.diagnose(dataset=testset, split="test")


def train_diagnose_teacher(
    model_client: adal.ModelClient,
    model_kwargs: Dict,
) -> Dict:

    trainset, valset, testset = load_datasets()

    adal_component = ObjectCountAdalComponent(model_client, model_kwargs)
    trainer = adal.Trainer(adaltask=adal_component)
    trainer.diagnose(dataset=trainset, split="train_teacher")
    trainer.diagnose(dataset=valset, split="val_teacher")
    trainer.diagnose(dataset=testset, split="test_teacher")


# You will answer a reasoning question. Think step by step and double-check each calculation you make. Pay close attention to any numerical quantities in the text, converting written numbers into their numerical equivalents. Additionally, re-verify your final answer before concluding. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.
# 0.98 val, 0.91 test
def train(
    train_batch_size=4,  # larger batch size is not that effective, probably because of llm's lost in the middle
    raw_shots: int = 0,
    bootstrap_shots: int = 1,
    max_steps=1,
    num_workers=4,
    strategy="random",
    optimization_order="sequential",
    debug=False,
    resume_from_ckpt=None,
    exclude_input_fields_from_bootstrap_demos=False,
):
    adal_component = ObjectCountAdalComponent(
        **gpt_3_model,
        teacher_model_config=gpt_4o_model,
        text_optimizer_model_config=gpt_4o_model,
        backward_engine_model_config=gpt_4o_model
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
        optimization_order=optimization_order,
        exclude_input_fields_from_bootstrap_demos=exclude_input_fields_from_bootstrap_demos,
    )
    print(trainer)

    train_dataset, val_dataset, test_dataset = load_datasets()
    trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        resume_from_ckpt=resume_from_ckpt,
    )


if __name__ == "__main__":

    train(
        debug=True,
        max_steps=12,
        strategy="constrained",
        exclude_input_fields_from_bootstrap_demos=True,
    )

    # train_diagnose(**gpt_3_model)
    # train_diagnose_teacher(**gpt_4o_model)
