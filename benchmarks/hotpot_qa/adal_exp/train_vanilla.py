from typing import Any, Callable, Dict, Tuple

import adalflow as adal
from adalflow.eval.answer_match_acc import AnswerMatchAcc
from adalflow.datasets.types import HotPotQAData

from benchmarks.hotpot_qa.adal_train import load_datasets
from benchmarks.hotpot_qa.adal_exp.build_vanilla_rag import VanillaRAG
from use_cases.config import gpt_3_model, gpt_4o_model


# TODO: look more into the loss function
# TODO: test LLM judge too.
class VallinaRAGAdal(adal.AdalComponent):
    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        backward_engine_model_config: Dict | None = None,
        teacher_model_config: Dict | None = None,
        text_optimizer_model_config: Dict | None = None,
    ):
        task = VanillaRAG(
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

    # tell the trainer how to call the task
    def prepare_task(self, sample: HotPotQAData) -> Tuple[Callable[..., Any], Dict]:
        if self.task.training:
            return self.task.forward, {"question": sample.question, "id": sample.id}
        else:
            return self.task.call, {"question": sample.question, "id": sample.id}

    # TODO: use two map fn to make the cde even simpler

    # eval mode: get the generator output, directly engage with the eval_fn
    def prepare_eval(self, sample: HotPotQAData, y_pred: adal.GeneratorOutput) -> float:
        y_label = ""
        if y_pred and y_pred.data and y_pred.data.answer:
            y_label = y_pred.data.answer
        return self.eval_fn, {"y": y_label, "y_gt": sample.answer}

    # train mode: get the loss and get the data from the full_response
    def prepare_loss(self, sample: HotPotQAData, pred: adal.Parameter):
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


# Note: diagnose is quite helpful, it helps you to quickly check if the evalfunction is the right metrics
# i checked the eval which does fuzzy match, and found some yes and Yes are not matched, then converted both strings to lower and
# the performances have gone up from 0.15 to 0.4
def train_diagnose(
    model_client: adal.ModelClient,
    model_kwargs: Dict,
) -> Dict:
    trainset, valset, testset = load_datasets()

    adal_component = VallinaRAGAdal(
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


def train(
    train_batch_size=4,  # larger batch size is not that effective, probably because of llm's lost in the middle
    raw_shots: int = 0,
    bootstrap_shots: int = 4,
    max_steps=1,
    num_workers=4,
    strategy="constrained",
    optimization_order="sequential",
    debug=False,
    resume_from_ckpt=None,
    exclude_input_fields_from_bootstrap_demos=True,
):
    adal_component = VallinaRAGAdal(
        **gpt_3_model,
        teacher_model_config=gpt_4o_model,
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
    from use_cases.config import gpt_3_model

    adal.setup_env()

    # task = VallinaRAGAdal(**gpt_3_model)
    # print(task)

    # train_diagnose(**gpt_3_model)

    # train: 0.15 before the evaluator converted to lower and 0.4 after the conversion
    # TODO: test debug mode
    train(
        debug=False,
        max_steps=12,
        # resume_from_ckpt="/Users/liyin/.adalflow/ckpt/ValinaRAGAdal/random_max_steps_12_7c091_run_1.json",
    )
    # random_max_steps_12_ecf16_run_9.json, demo only, val 0.6 to 0.68,  test: 0.58-0.61
    # random_max_steps_12_7c091_run_1.json,  prompt + demo, 0.58 -0.62, test: 0.55 - 0.58
    # resume from random_max_steps_12_7c091_run_1.json

    # demo only, no input, 4 shots, 0.58-> 0.62, VallinaRAGAdal/constrained_max_steps_12_b0a37_run_1.json
    # this is the same as dspy's 20shots, because dspy does not use the weighted sampling
