from typing import Any, Callable, Dict, Tuple
import adalflow as adal
from adalflow.eval.answer_match_acc import AnswerMatchAcc
from benchmarks.hotpot_qa.adal_exp.build_vanila_rag import VanilaRAG
from use_cases.config import gpt_3_model, gpt_4o_model
from adalflow.datasets.types import HotPotQAData

from benchmarks.hotpot_qa.adal_train import load_datasets


# TODO: look more into the loss function
class ValinaRAGAdal(adal.AdalComponent):
    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        backward_engine_model_config: Dict | None = None,
        teacher_model_config: Dict | None = None,
        text_optimizer_model_config: Dict | None = None,
    ):
        task = VanilaRAG(
            model_client=model_client,
            model_kwargs=model_kwargs,
            passages_per_hop=3,
        )
        eval_fn = AnswerMatchAcc(type="fuzzy_match").compute_single_item
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn, eval_fn_desc="fuzzy_match: 1 if str(y) in str(y_gt) else 0"
        )
        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            backward_engine_model_config=backward_engine_model_config,
            teacher_model_config=teacher_model_config,
            text_optimizer_model_config=text_optimizer_model_config,
        )

    def handle_one_task_sample(
        self, sample: HotPotQAData
    ) -> Tuple[Callable[..., Any], Dict]:
        if self.task.training:  # TODO: make the components more clear
            return self.task.forward, {"question": sample.question, "id": sample.id}
        else:
            return self.task.call, {"question": sample.question, "id": sample.id}

    # TODO: use two map fn to make the cde even simpler

    # eval mode: get the generator output, directly engage with the eval_fn
    def evaluate_one_sample(
        self, sample: HotPotQAData, y_pred: adal.GeneratorOutput
    ) -> float:
        y_label = ""
        if y_pred and y_pred.data and y_pred.data.answer:
            y_label = y_pred.data.answer
        return self.eval_fn(y=y_label, y_gt=sample.answer)

    # train mode: get the loss and get the data from the full_response
    def handle_one_loss_sample(self, sample: HotPotQAData, pred: adal.Parameter):
        # prepare gt parameter
        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.answer,
            eval_input=sample.answer,
            requires_opt=False,
        )

        # pred's full_response is the output of the task pipeline which is GeneratorOutput
        pred.eval_input = (
            pred.full_response.data.answer
            if pred.full_response
            and pred.full_response.data
            and pred.full_response.data.answer
            else ""
        )
        return self.loss_fn, {"kwargs": {"y": pred, "y_gt": y_gt}}


def train_diagnose(
    model_client: adal.ModelClient,
    model_kwargs: Dict,
) -> Dict:

    trainset, valset, testset = load_datasets()

    adal_component = ValinaRAGAdal(
        model_client,
        model_kwargs,
        backward_engine_model_config=gpt_4o_model,
        teacher_model_config=gpt_3_model,
        text_optimizer_model_config=gpt_3_model,
    )
    trainer = adal.Trainer(adaltask=adal_component)
    trainer.diagnose(dataset=trainset, split="train")
    # trainer.diagnose(dataset=valset, split="val")
    # trainer.diagnose(dataset=testset, split="test")


if __name__ == "__main__":
    from use_cases.config import gpt_3_model

    adal.setup_env()

    task = ValinaRAGAdal(**gpt_3_model)
    print(task)

    train_diagnose(**gpt_3_model)
