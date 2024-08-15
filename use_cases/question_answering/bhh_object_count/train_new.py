from adalflow.optim.trainer.trainer import Trainer

from use_cases.question_answering.bhh_object_count.task import (
    ObjectCountTaskPipeline,
)

from use_cases.question_answering.bhh_object_count.config import (
    gpt_3_model,
    gpt_4o_model,
)

from typing import Dict, Optional
import adalflow as adal

from adalflow.datasets.types import Example
from adalflow.eval.answer_match_acc import AnswerMatchAcc


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

    def handle_one_task_sample(self, sample: Example):
        return self.task.call, {"question": sample.question, "id": sample.id}

    def evaluate_one_sample(
        self, sample: Example, y_pred: adal.GeneratorOutput
    ) -> float:
        y_label = -1
        if y_pred and y_pred.data:
            y_label = y_pred.data
        return self.eval_fn(y=y_label, y_gt=sample.answer)

    # needed for training
    def handle_one_loss_sample(self, sample: Example, pred: adal.Parameter):
        # prepare gt parameter
        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.answer,
            eval_input=sample.answer,
            requires_opt=False,
        )

        # pred's full_response is the output of the task pipeline which is GeneratorOutput
        pred.eval_input = pred.full_response.data
        return self.loss_fn, {"kwargs": {"y": pred, "y_gt": y_gt}}

    def configure_backward_engine(self):
        super().configure_backward_engine_helper(**self.backward_engine_model_config)

    def configure_teacher_generator(self):
        super().configure_teacher_generator_helper(**self.teacher_model_config)

    def configure_optimizers(
        self,
    ):  # TODO: train the text optimizer and the demo optimizer at the same time
        to = super().configure_text_optimizer_helper(**self.text_optimizer_model_config)
        do = super().configure_demo_optimizer_helper()
        return to + do


# TODO: make the train diagnose on the student model and the teacher model automatcally
# in the trainer
def train_diagnose(
    model_client: adal.ModelClient,
    model_kwargs: Dict,
) -> Dict:
    from use_cases.question_answering.bhh_object_count.data import load_datasets

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
    from use_cases.question_answering.bhh_object_count.data import load_datasets

    trainset, valset, testset = load_datasets()

    adal_component = ObjectCountAdalComponent(model_client, model_kwargs)
    trainer = adal.Trainer(adaltask=adal_component)
    trainer.diagnose(dataset=trainset, split="train_teacher")
    trainer.diagnose(dataset=valset, split="val_teacher")
    trainer.diagnose(dataset=testset, split="test_teacher")


from use_cases.question_answering.bhh_object_count.data import load_datasets


# You will answer a reasoning question. Think step by step and double-check each calculation you make. Pay close attention to any numerical quantities in the text, converting written numbers into their numerical equivalents. Additionally, re-verify your final answer before concluding. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.
# 0.98 val, 0.91 test
def train(
    train_batch_size=4,  # larger batch size is not that effective, probably because of llm's lost in the middle
    raw_shots: int = 0,
    bootstrap_shots: int = 1,
    max_steps=1,
    num_workers=4,
    strategy="random",
    debug=False,
):
    adal_component = ObjectCountAdalComponent(
        **gpt_3_model,
        teacher_model_config=gpt_3_model,
        text_optimizer_model_config=gpt_4o_model,
        backward_engine_model_config=gpt_4o_model
    )
    print(adal_component)
    trainer = Trainer(
        train_batch_size=train_batch_size,
        strategy=strategy,
        max_steps=max_steps,
        num_workers=num_workers,
        adaltask=adal_component,
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

    train(
        debug=False, max_steps=24, strategy="constrained"
    )  # TODO: few-shot constraint

    # train_diagnose(**gpt_3_model)
    # train_diagnose_teacher(**gpt_4o_model)
