from typing import Dict
import adalflow as adal
from use_cases.classification.trec_task_structured_output import (
    TRECClassifierStructuredOutput,
)
from use_cases.classification.trec_task_string_output import (
    TRECClassifierStringOutput,
)
from use_cases.classification.data import TRECExtendedData

from adalflow.eval.answer_match_acc import AnswerMatchAcc


class TrecClassifierAdal(adal.AdalComponent):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        task = TRECClassifierStructuredOutput(model_client, model_kwargs)
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        super().__init__(task=task, eval_fn=eval_fn)

    def handle_one_task_sample(self, sample: TRECExtendedData):
        return self.task.call, {"question": sample.question, "id": sample.id}

    def evaluate_one_sample(
        self, sample: TRECExtendedData, y_pred: adal.GeneratorOutput
    ) -> float:
        y_label = -1
        if y_pred and y_pred.data is not None and y_pred.data.class_name is not None:
            y_label = y_pred.data.class_name
        return self.eval_fn(y_label, sample.class_name)


def diagnose(
    model_client: adal.ModelClient,
    model_kwargs: Dict,
    is_teacher: bool = False,
) -> Dict:
    from use_cases.classification.data import load_datasets

    trainset, valset, testset = load_datasets()

    adal_component = TrecClassifierAdal(model_client, model_kwargs)
    trainer = adal.Trainer(adaltask=adal_component)
    trainer.diagnose(
        dataset=trainset, split="train" if not is_teacher else "train_teacher"
    )
    trainer.diagnose(dataset=valset, split="val" if not is_teacher else "val_teacher")
    trainer.diagnose(
        dataset=testset, split="test" if not is_teacher else "test_teacher"
    )


class TrecClassifierStringOutputAdal(adal.AdalComponent):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        task = TRECClassifierStringOutput(
            model_client, model_kwargs
        )  # update the the different task
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        super().__init__(task=task, eval_fn=eval_fn)

    def handle_one_task_sample(self, sample: TRECExtendedData):
        return self.task.call, {"question": sample.question, "id": sample.id}

    def evaluate_one_sample(
        self, sample: TRECExtendedData, y_pred: adal.GeneratorOutput
    ) -> float:
        y_label = -1
        if y_pred and y_pred.data is not None:  # use different output format
            y_label = y_pred.data
        return self.eval_fn(y_label, sample.class_index)


def diagnose_string_output(
    model_client: adal.ModelClient,
    model_kwargs: Dict,
) -> Dict:
    from use_cases.classification.data import load_datasets

    trainset, valset, testset = load_datasets()

    adal_component = TrecClassifierStringOutputAdal(model_client, model_kwargs)
    trainer = adal.Trainer(adaltask=adal_component)
    trainer.diagnose(dataset=trainset, split="train")
    trainer.diagnose(dataset=valset, split="val")
    trainer.diagnose(dataset=testset, split="test")


if __name__ == "__main__":
    from use_cases.config import (
        gpt_4o_model,
    )

    # diagnose(**gpt_3_model)  # train: 0.692 # test:0.77 # val:0.694
    # use class name and ask it to select: test: 82.64% # val: 69.4% # train: 67.5%Final
    # diagnose_string_output(**gpt_3_model)  # train: 0.7 # test: 0.764 # val: 0.7

    diagnose(
        **gpt_4o_model, is_teacher=True
    )  # train_teacher: 0.767 # test_teacher: 0.82 # val_teacher: 0.75

    # teacher class: train: 77.5%, # val: 77.78% # test: 86.11%
    # optimized teacher: train: 77.8%, # val: 77.78% # test: 84.03% (optimized prompt might not apply to another model)

    # there is no point of using bootstrap if the teacher is not better than optimized zero-shot.
