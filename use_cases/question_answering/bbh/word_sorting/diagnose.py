from typing import Dict
import adalflow as adal
from use_cases.question_answering.bbh.word_sorting.task import (
    QuestionAnswerTaskPipeline,
)

from adalflow.datasets.types import Example
from adalflow.eval.llm_as_judge import DefaultLLMJudge
from use_cases.question_answering.bbh.data import load_datasets
from use_cases.config import gpt_3_model


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
    ):
        task = QuestionAnswerTaskPipeline(model_client, model_kwargs)

        llm_judge = DefaultLLMJudge(
            **llm_judge_model_config,
            output_type="float",
            jugement_query=judgement_query,
            use_cache=True,
        )
        eval_fn = llm_judge.call
        # eval_fn = lambda question, gt_answer, pred_answer: 1
        super().__init__(task=task, eval_fn=eval_fn)

    def handle_one_task_sample(self, sample: Example):
        return self.task.call, {"question": sample.question, "id": sample.id}

    def evaluate_one_sample(
        self, sample: Example, y_pred: adal.GeneratorOutput
    ) -> float:
        y_label = ""
        if y_pred and y_pred.data:
            y_label = y_pred.data
        return self.eval_fn(
            question=sample.question, gt_answer=sample.answer, pred_answer=y_label
        )


def evaluate_one_sample():

    trainset, valset, testset = load_datasets(task_name="BBH_word_sorting")
    adal_component = WordSortingAdalComponent(
        **gpt_3_model, llm_judge_model_config=gpt_3_model
    )
    example = trainset[1]
    call, kwargs = adal_component.handle_one_task_sample(example)
    output = call(**kwargs)
    print(f"output: {output}")
    print(f"trainset[0]: {example}")
    score = adal_component.evaluate_one_sample(example, output)
    print(score)


def diagnose(
    model_client: adal.ModelClient,
    model_kwargs: Dict,
) -> Dict:

    trainset, valset, testset = load_datasets(task_name="BBH_word_sorting")
    adal_component = WordSortingAdalComponent(
        model_client, model_kwargs, llm_judge_model_config=gpt_3_model
    )
    trainer = adal.Trainer(adaltask=adal_component, num_workers=4)
    trainer.diagnose(dataset=trainset, split="train")
    trainer.diagnose(dataset=valset, split="val")
    trainer.diagnose(dataset=testset, split="test")


if __name__ == "__main__":
    from use_cases.config import (
        gpt_3_model,
    )

    # evaluate_one_sample()
    diagnose(**gpt_3_model)
