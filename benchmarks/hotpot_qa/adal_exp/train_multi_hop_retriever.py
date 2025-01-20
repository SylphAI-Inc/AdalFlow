from typing import Any, Callable, Dict, Tuple, List

import adalflow as adal
from adalflow.eval.retriever_recall import RetrieverEvaluator
from adalflow.eval.answer_match_acc import AnswerMatchAcc
from adalflow.datasets.types import HotPotQAData
from benchmarks.hotpot_qa.config import load_datasets

from benchmarks.hotpot_qa.adal_exp.build_multi_hop_rag import (
    MultiHopRetriever,
)
from use_cases.config import gpt_3_model, gpt_4o_model
from adalflow.utils import printc


def retriever_recall(y: List[str], y_gt: List[str]) -> float:
    return RetrieverEvaluator().compute_single_item(y, y_gt)["recall"]


def retriever_precision(y: List[str], y_gt: List[str]) -> float:
    return RetrieverEvaluator().compute_single_item(y, y_gt)["precision"]


def retriever_query_f1(y: str, y_gt: str) -> float:
    evaluator = AnswerMatchAcc(type="f1_score")
    score = evaluator.compute_single_item(y, y_gt)

    return score


class MultiHopRetrieverAdal(adal.AdalComponent):
    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        backward_engine_model_config: Dict | None = None,
        teacher_model_config: Dict | None = None,
        text_optimizer_model_config: Dict | None = None,
    ):
        task = MultiHopRetriever(
            model_client=model_client,
            model_kwargs=model_kwargs,
            passages_per_hop=2,
            max_hops=2,
        )
        eval_fn = retriever_recall
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="recall: len(y_gt.intersection(y)) / len(y_gt)",
        )
        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            backward_engine_model_config=backward_engine_model_config,
            teacher_model_config=teacher_model_config,
            text_optimizer_model_config=text_optimizer_model_config,
        )

    def prepare_task(self, sample: HotPotQAData) -> Tuple[Callable[..., Any], Dict]:
        if self.task.training:
            return self.task.forward, {"input": sample.question, "id": sample.id}
        else:
            return self.task.call, {"input": sample.question, "id": sample.id}

    def prepare_eval(self, sample: HotPotQAData, y_pred: adal.RetrieverOutput) -> float:
        if isinstance(y_pred, adal.Parameter):
            raise ValueError("y_pred is not a RetrieverOutput")
        documents = y_pred.documents
        y_pred_titles = []
        for doc in documents:
            title, content = doc.split("|")
            y_pred_titles.append(title)

        return self.eval_fn, {
            "y": y_pred_titles,
            "y_gt": list(sample.gold_titles),
        }

    def prepare_loss(self, sample: HotPotQAData, pred: adal.Parameter):
        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.gold_titles,
            eval_input=list(sample.gold_titles),
            requires_opt=False,
        )

        pred_titles = []
        for doc in pred.data.documents:
            title, content = doc.split("|")
            pred_titles.append(title)

        pred.eval_input = pred_titles
        return self.loss_fn, {
            "kwargs": {"y": pred, "y_gt": y_gt},
            "id": sample.id,
            "gt": y_gt.data,
        }


# 1. test the eval and the loss use different metrics
class MultiHopRetriever2Adal(adal.AdalComponent):
    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        backward_engine_model_config: Dict | None = None,
        teacher_model_config: Dict | None = None,
        text_optimizer_model_config: Dict | None = None,
    ):
        task = MultiHopRetriever(
            model_client=model_client,
            model_kwargs=model_kwargs,
            passages_per_hop=2,
            max_hops=2,
        )
        eval_fn = retriever_query_f1
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="precision: overlap of words between gt and prediction (queries). Only evaluate the generated queries from the generator. The multiple queries are joiend together by ',' to evaluate over the overlap on words.",
        )
        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            backward_engine_model_config=backward_engine_model_config,
            teacher_model_config=teacher_model_config,
            text_optimizer_model_config=text_optimizer_model_config,
        )
        self.eval_retriever_recall = retriever_recall

    def prepare_task(self, sample: HotPotQAData) -> Tuple[Callable[..., Any], Dict]:
        if self.task.training:
            return self.task.forward2, {"input": sample.question, "id": sample.id}
        else:
            return self.task.call2, {"input": sample.question, "id": sample.id}

    def prepare_eval(self, sample: HotPotQAData, y_pred: any) -> float:
        if isinstance(y_pred, adal.Parameter):
            raise ValueError("y_pred is not a RetrieverOutput")

        y_gt = ", ".join(sample.gold_titles)
        # for doc in documents:
        #     title, content = doc.split("|")
        #     y_pred_titles.append(title)

        printc(f"y_gt: {y_gt}, pred: {y_pred}")

        return self.eval_fn, {
            "y": y_pred.data,
            "y_gt": y_gt,
        }

    def prepare_loss(self, sample: HotPotQAData, pred: adal.Parameter):

        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.gold_titles,
            eval_input=", ".join(sample.gold_titles),
            requires_opt=False,
        )

        pred.eval_input = pred.data.data

        printc(f"y_gt 1: {sample.gold_titles}, pred 1: {pred.eval_input}")

        return self.loss_fn, {
            "kwargs": {"y": pred, "y_gt": y_gt},
            "id": sample.id,
            "gt": y_gt.data,
        }


from adalflow.core.generator import BackwardPassSetup


def train_diagnose(
    model_client: adal.ModelClient,
    model_kwargs: Dict,
) -> Dict:

    trainset, valset, testset = load_datasets()

    adal_component = MultiHopRetrieverAdal(
        model_client,
        model_kwargs,
        backward_engine_model_config=gpt_4o_model,
        teacher_model_config=gpt_3_model,
        text_optimizer_model_config=gpt_3_model,
    )
    trainer = adal.Trainer(adaltask=adal_component)
    # trainer.diagnose(dataset=trainset, split="train")  # 0.69 recall
    # trainer.diagnose(dataset=valset, split="val")  # 0.675 recall
    trainer.diagnose(dataset=testset, split="test")  # 0.71 (0.665)


def train(
    train_batch_size=4,  # larger batch size is not that effective, probably because of llm's lost in the middle
    raw_shots: int = 1,
    bootstrap_shots: int = 1,
    max_steps=1,
    num_workers=10,
    strategy="constrained",
    optimization_order="sequential",
    debug=False,
    resume_from_ckpt=None,
    exclude_input_fields_from_bootstrap_demos=True,
    seed=None,
    tg: bool = False,
    max_proposals_per_step: int = 5,
):
    adal_component = MultiHopRetrieverAdal(
        **gpt_3_model,
        teacher_model_config=gpt_4o_model,
        text_optimizer_model_config=gpt_4o_model,  # gpt3.5 is not enough to be used as a good optimizer, it struggles for long contenxt
        backward_engine_model_config=gpt_4o_model,
    )
    backward_pass_setup = None
    if tg:
        backward_pass_setup = BackwardPassSetup(
            all_pred_at_once=False,
            compute_grad_for_errors_only=False,
        )
    # print(adal_component)
    trainer = adal.Trainer(
        train_batch_size=train_batch_size,
        adaltask=adal_component,
        strategy=strategy,
        max_steps=max_steps,
        num_workers=num_workers,
        raw_shots=raw_shots,
        bootstrap_shots=bootstrap_shots,
        debug=debug,
        weighted_sampling=False,
        optimization_order=optimization_order,
        exclude_input_fields_from_bootstrap_demos=exclude_input_fields_from_bootstrap_demos,
        sequential_order=["text", "demo"],
        max_proposals_per_step=max_proposals_per_step,
        backward_pass_setup=backward_pass_setup,
    )
    trainer.set_random_seed(seed)
    print(trainer)

    train_dataset, val_dataset, test_dataset = load_datasets()
    # val_dataset = val_dataset[:20]
    ckpt, _ = trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        resume_from_ckpt=resume_from_ckpt,
    )
    return ckpt


if __name__ == "__main__":
    from use_cases.config import gpt_3_model

    # log = adal.get_logger(level="DEBUG", enable_console=False)

    adal.setup_env()

    import json

    import random

    random.seed(2025)

    adal.setup_env()

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--strategy", type=str, default="constrained")
    parser.add_argument("--use_tg", action="store_false")
    parser.add_argument("--max_proposals_per_step", type=int, default=5)
    parser.add_argument(
        "output_path", nargs="?", help="File path to save the checkpoint"
    )

    args = parser.parse_args()

    set_strategy = args.strategy
    set_output_path = args.output_path
    use_tg = args.use_tg
    max_proposals_per_step = args.max_proposals_per_step

    # task = MultiHopRAGAdal(**gpt_3_model)
    # print(task)

    # train_diagnose(**gpt_3_model)
    # exit()

    # train: 0.15 before the evaluator converted to lower and 0.4 after the conversion
    ckpt = train(
        debug=True,
        max_steps=12,
        seed=2025,  # pass the numpy seed
        tg=use_tg,
        strategy=set_strategy,
        max_proposals_per_step=max_proposals_per_step,
        exclude_input_fields_from_bootstrap_demos=True,
        # resume_from_ckpt="/Users/liyin/.adalflow/ckpt/MultiHopRetrieverAdal/constrained_max_steps_12_945bd_run_1.json",
        # resume_from_ckpt="/Users/liyin/.adalflow/ckpt/MultiHopRetrieverAdal/constrained_max_steps_12_d7043_run_1.json",
    )
    print(f"ckpt: {ckpt}")
    if set_output_path:
        with open(set_output_path, "w") as f:
            json.dump({"ckpt": ckpt}, f)
        print(f"Checkpoint saved to {set_output_path}")
    else:
        print("No file path provided for saving the checkpoint.")

    #
