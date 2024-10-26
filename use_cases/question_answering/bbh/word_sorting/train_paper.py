from typing import Dict
import adalflow as adal
from use_cases.question_answering.bbh.word_sorting.task import (
    QuestionAnswerTaskPipeline,
)

from adalflow.datasets.types import Example
from adalflow.eval.llm_as_judge import DefaultLLMJudge
from use_cases.question_answering.bbh.data import load_datasets
from use_cases.config import gpt_3_model, gpt_4o_model


judgement_query = r"""Does the predicted answer match with the ground truth answer? PLEASE Ignore the difference of separators between words.
Say True if it matches, False if not.
Example:
Question: Sort the following words alphabetically: List: syndrome therefrom
Ground truth answer: syndrome therefrom
Predicted answer: syndrome, therefrom
Answer: True"""


class WordSortingAdalComponent(adal.AdalComponent):
    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        llm_judge_model_config: Dict,
        backward_engine_model_config: Dict = None,
        teacher_model_config: Dict = None,
        text_optimizer_model_config: Dict = None,
    ):
        task = QuestionAnswerTaskPipeline(model_client, model_kwargs)

        llm_judge = DefaultLLMJudge(
            **llm_judge_model_config,
            output_type="float",
            jugement_query=judgement_query,
            use_cache=True,
        )
        eval_fn = llm_judge.call
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="exact_match: 1 if str(y) == str(y_gt) else 0",
        )
        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            backward_engine_model_config=backward_engine_model_config,
            teacher_model_config=teacher_model_config,
            text_optimizer_model_config=text_optimizer_model_config,
        )

    def prepare_task(self, sample: Example):
        return self.task.call, {"question": sample.question, "id": sample.id}

    def prepare_eval(self, sample: Example, y_pred: adal.GeneratorOutput) -> float:
        y_label = ""
        if (
            y_pred is not None and y_pred.data is not None
        ):  # if y_pred and y_pred.data: might introduce bug when the data is 0
            y_label = y_pred.data

        return self.eval_fn, {
            "question": sample.question,
            "gt_answer": sample.answer,
            "pred_answer": y_label,
        }

    def prepare_loss(self, sample: Example, pred: adal.Parameter):
        # prepare gt parameter
        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.answer,
            eval_input=sample.answer,
            requires_opt=False,
        )
        pred.eval_input = pred.full_response.data  # processed
        question_param = adal.Parameter(
            name="question",
            data=sample.question,
            eval_input=sample.question,
            requires_opt=False,
        )

        return self.loss_fn, {
            "kwargs": {
                "pred_answer": pred,
                "gt_answer": y_gt,
                "question": question_param,
            }
        }


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
    adal_component = WordSortingAdalComponent(
        **gpt_3_model,
        teacher_model_config=gpt_4o_model,
        text_optimizer_model_config=gpt_4o_model,
        backward_engine_model_config=gpt_4o_model,
        llm_judge_model_config=gpt_3_model,
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

    train_dataset, val_dataset, test_dataset = load_datasets(task_name="word_sorting")
    trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        resume_from_ckpt=resume_from_ckpt,
    )


if __name__ == "__main__":

    train(
        debug=False,
        max_steps=10,
        strategy="constrained",
        exclude_input_fields_from_bootstrap_demos=False,
        # resume_from_ckpt="constrained_max_steps_12_7dc6a_run_2.json",
    )
