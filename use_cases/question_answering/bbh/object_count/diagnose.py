from typing import Dict
import adalflow as adal
from use_cases.question_answering.bbh.object_count.task import ObjectCountTaskPipeline

from adalflow.datasets.types import Example
from adalflow.eval.answer_match_acc import AnswerMatchAcc


class ObjectCountAdalComponent(adal.AdalComponent):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        task = ObjectCountTaskPipeline(model_client, model_kwargs)
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        super().__init__(task=task, eval_fn=eval_fn)

    def prepare_task(self, sample: Example):
        return self.task.call, {"question": sample.question, "id": sample.id}

    def prepare_eval(self, sample: Example, y_pred: adal.GeneratorOutput) -> float:
        y_label = -1
        if y_pred and y_pred.data:
            y_label = y_pred.data
        return self.eval_fn, {"y": y_label, "y_gt": sample.answer}


def diagnose(
    model_client: adal.ModelClient,
    model_kwargs: Dict,
) -> Dict:
    from use_cases.question_answering.bbh.data import load_datasets

    trainset, valset, testset = load_datasets()
    # use max_samples=10 to test the code

    adal_component = ObjectCountAdalComponent(model_client, model_kwargs)
    trainer = adal.Trainer(adaltask=adal_component)
    trainer.diagnose(dataset=trainset, split="train")
    trainer.diagnose(dataset=valset, split="val")
    trainer.diagnose(dataset=testset, split="test")


if __name__ == "__main__":
    from use_cases.config import (
        gpt_3_model,
    )

    # from use_cases.question_answering.bhh_object_count.prepare_trainer import (
    #     TGDWithEvalFnLoss,
    # )

    # from adalflow.optim.trainer.trainer import Trainer

    # trainset, valset, testset = load_datasets(max_samples=10)
    # adaltask = TGDWithEvalFnLoss(
    #     task_model_config=llama3_model,
    #     backward_engine_model_config=llama3_model,
    #     optimizer_model_config=llama3_model,
    # )

    # trainer = Trainer(adaltask=adaltask)
    # diagnose = trainer.diagnose(train_dataset=trainset)
    # print(diagnose)

    # Diagnostic results run on trainer set, with all inputs and outputs tracked in ckpt/TGDWithEvalFnLoss/llm_counter_call

    diagnose(**gpt_3_model)
