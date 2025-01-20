from typing import Any, Callable, Dict, Tuple

import adalflow as adal
from adalflow.eval.answer_match_acc import AnswerMatchAcc
from adalflow.datasets.types import HotPotQAData

from benchmarks.hotpot_qa.config import load_datasets
from benchmarks.hotpot_qa.adal_exp.build_multi_hop_rag import AgenticRAG
from use_cases.config import gpt_3_model, gpt_4o_model
from adalflow.utils import printc


# TODO: look more into the loss function
# TODO: test LLM judge too.

from adalflow.components.agent.react import ReActOutput


class AgenticRAGAdal(adal.AdalComponent):
    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        backward_engine_model_config: Dict | None = None,
        teacher_model_config: Dict | None = None,
        text_optimizer_model_config: Dict | None = None,
    ):
        task = AgenticRAG(
            model_client=model_client,
            model_kwargs=model_kwargs,
        )
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        loss_eval_fn = AnswerMatchAcc(type="f1_score").compute_single_item
        # eval_fn = AnswerMatchAcc(type="f1_score").compute_single_item
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=loss_eval_fn,
            eval_fn_desc="exact_match: 1 if str(y_gt) == str(y) else 0",
        )
        # eval_fn = f1_score  # 0.38 (hand crafted the finish, exat match 0.25)

        # loss_fn = adal.EvalFnToTextLoss(
        #     eval_fn=eval_fn, eval_fn_desc="Computes the overlaps between y and y_gt"
        # )
        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_eval_fn=loss_eval_fn,
            loss_fn=loss_fn,
            backward_engine_model_config=backward_engine_model_config,
            teacher_model_config=teacher_model_config,
            text_optimizer_model_config=text_optimizer_model_config,
        )

    # tell the trainer how to call the task
    def prepare_task(self, sample: HotPotQAData) -> Tuple[Callable[..., Any], Dict]:
        if self.task.training:
            return self.task.forward, {"input": sample.question, "id": sample.id}
        else:
            # print("eval mode")
            return self.task.call, {"input": sample.question, "id": sample.id}

    # TODO: use two map fn to make the cde even simpler

    # eval mode: get the generator output, directly engage with the eval_fn
    def prepare_eval(self, sample: HotPotQAData, y_pred: ReActOutput) -> float:

        y_label = y_pred.answer if isinstance(y_pred, ReActOutput) else y_pred

        printc(
            f"eval y_label: {y_label}, y_gt: {sample.answer}, self.eval_fn: {self.eval_fn(y_label, sample.answer)}"
        )

        return self.eval_fn, {"y": y_label, "y_gt": sample.answer}

    def prepare_loss_eval(self, sample: HotPotQAData, y_pred: ReActOutput) -> float:
        y_label = y_pred.answer if isinstance(y_pred, ReActOutput) else y_pred
        printc(
            f"loss eval y_label: {y_label}, y_gt: {sample.answer}, self.eval_fn: {self.loss_eval_fn(y_label, sample.answer)}"
        )
        return self.loss_eval_fn, {"y": y_label, "y_gt": sample.answer}

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
        # pred.eval_input = (
        #     pred.data[-1].observation if pred.data and pred.data[-1] else ""
        # )
        printc(f"pred data: {pred.data}, gt: {sample.answer}")
        pred.eval_input = pred.data if pred.data else ""
        # pred.eval_input = (
        #     pred.data[-1].observation if pred.data and pred.data[-1] else ""
        # )
        # printc(f"loss eval_input: {pred.eval_input}")
        return self.loss_fn, {
            "kwargs": {"y": pred, "y_gt": y_gt},
            "id": sample.id,
            "gt": y_gt.eval_input,
            "input": {"question": sample.question},
        }


# Note: diagnose is quite helpful, it helps you to quickly check if the evalfunction is the right metrics
# i checked the eval which does fuzzy match, and found some yes and Yes are not matched, then converted both strings to lower and
# the performances have gone up from 0.15 to 0.4
def train_diagnose(
    model_client: adal.ModelClient,
    model_kwargs: Dict,
) -> Dict:

    trainset, valset, testset = load_datasets()

    adal_component = AgenticRAGAdal(
        model_client,
        model_kwargs,
        backward_engine_model_config=gpt_4o_model,
        teacher_model_config=gpt_3_model,
        text_optimizer_model_config=gpt_3_model,
    )
    trainset = trainset[:5]
    trainer = adal.Trainer(adaltask=adal_component)
    trainer.diagnose(dataset=trainset, split="train")
    # trainer.diagnose(dataset=valset, split="val")
    # trainer.diagnose(dataset=testset, split="test")


from adalflow.core.generator import BackwardPassSetup


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
    seed=None,
    tg: bool = False,
    max_proposals_per_step: int = 5,
):
    adal_component = AgenticRAGAdal(
        **gpt_3_model,
        teacher_model_config=gpt_4o_model,
        text_optimizer_model_config=gpt_4o_model,  # gpt3.5 is not enough to be used as a good optimizer, it struggles for long contenxt
        backward_engine_model_config=gpt_4o_model,
    )
    print(adal_component)
    backward_pass_setup = None
    if tg:
        backward_pass_setup = BackwardPassSetup(
            all_pred_at_once=False,
            compute_grad_for_errors_only=False,
        )
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
        sequential_order=["text", "demo"],
        max_proposals_per_step=max_proposals_per_step,
        backward_pass_setup=backward_pass_setup,
    )
    trainer.set_random_seed(seed)
    print(trainer)

    train_dataset, val_dataset, test_dataset = load_datasets()
    # train_dataset = train_dataset[:40]
    # val_dataset = val_dataset[:40]
    # test_dataset = test_dataset[:40]

    ckpt, _ = trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        resume_from_ckpt=resume_from_ckpt,
    )
    return ckpt


if __name__ == "__main__":
    from use_cases.config import gpt_3_model

    log = adal.get_logger(level="DEBUG", enable_console=False)

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

    ckpt = train(
        debug=False,
        max_steps=12,
        seed=2025,
        tg=use_tg,
        strategy=set_strategy,
        max_proposals_per_step=max_proposals_per_step,
        # resume_from_ckpt="/Users/liyin/.adalflow/ckpt/AgenticRAGAdal/constrained_max_steps_12_387b2_run_1.json",
        # resume_from_ckpt="/Users/liyin/.adalflow/ckpt/AgenticRAGAdal/constrained_max_steps_4_dca7e_run_1.json",
    )
    print(f"ckpt: {ckpt}")
    if set_output_path:
        with open(set_output_path, "w") as f:
            json.dump({"ckpt": ckpt}, f)
        print(f"Checkpoint saved to {set_output_path}")
    else:
        print("No file path provided for saving the checkpoint.")

    # 0.68 on val without training, 0.74on the second step. 0.84 test
    # /Users/liyin/.adalflow/ckpt/AgenticRAGAdal/constrained_max_steps_2_029cb_run_1.json
    # 0.7, 0.72 /Users/liyin/.adalflow/ckpt/AgenticRAGAdal/constrained_max_steps_2_b7523_run_1.json
    # 208.085706949234s, 2 steps, maximum 4 steps allow for an agent.
    # 0.72->0.74, 4 steps, 366s, /Users/liyin/.adalflow/ckpt/AgenticRAGAdal/constrained_max_steps_4_dca7e_run_1.json [Already faster, still lots to optimize]

    # 1246s, 12 steps, 0.8 val, /Users/liyin/.adalflow/ckpt/AgenticRAGAdal/constrained_max_steps_12_defe7_run_1.json
    # 2149s, both gradients, 0.68 -> 0.78 /Users/liyin/.adalflow/ckpt/AgenticRAGAdal/constrained_max_steps_12_8a24a_run_1.json
    # /Users/liyin/.adalflow/ckpt/AgenticRAGAdal/constrained_max_steps_12_cdcb5_run_1.json 1728 s, 0.8
    # /Users/liyin/.adalflow/ckpt/AgenticRAGAdal/constrained_max_steps_12_735a7_run_1.json 0.58 -> 0.68 (separate gradients)  "pass": 17,
    #       "fail": 35
