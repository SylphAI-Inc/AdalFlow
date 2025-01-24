from typing import Any, Callable, Dict, Tuple
import adalflow as adal
from use_cases.classification.trec_task_structured_output import (
    TRECClassifierStructuredOutput,
)

from use_cases.classification.data import load_datasets, TRECExtendedData

from adalflow.eval.answer_match_acc import AnswerMatchAcc
from use_cases.config import (
    gpt_3_model,
    gpt_4o_model,
)
from adalflow.core.generator import BackwardPassSetup


class TrecClassifierAdal(adal.AdalComponent):
    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        teacher_model_config: Dict,
        backward_engine_model_config: Dict,
        text_optimizer_model_config: Dict,
    ):
        task = TRECClassifierStructuredOutput(model_client, model_kwargs)
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="exact_match: 1 if str(y) == str(y_gt) else 0. When the LLM prediction failed with format parsing which results with errors, we set y_pred = -1",
        )
        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            backward_engine_model_config=backward_engine_model_config,
            text_optimizer_model_config=text_optimizer_model_config,
            teacher_model_config=teacher_model_config,
        )

    def prepare_task(self, sample: TRECExtendedData):
        return self.task.call, {"question": sample.question, "id": sample.id}

    def prepare_eval(
        self, sample: TRECExtendedData, y_pred: adal.GeneratorOutput
    ) -> float:
        y_label = -1
        if y_pred and y_pred.data is not None and y_pred.data.class_name is not None:
            y_label = y_pred.data.class_name
        return self.eval_fn, {"y": y_label, "y_gt": sample.class_name}

    def prepare_loss(
        self, sample: TRECExtendedData, y_pred: adal.Parameter, *args, **kwargs
    ) -> Tuple[Callable[..., Any], Dict]:
        full_response = y_pred.data
        y_label = -1  # default value for failed prediction
        if (
            full_response
            and full_response.data is not None
            and full_response.data.class_name is not None
        ):
            y_label = full_response.data.class_name

        y_pred.eval_input = y_label
        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.class_name,
            eval_input=sample.class_name,
            requires_opt=False,
        )
        return self.loss_fn, {
            "kwargs": {"y": y_pred, "y_gt": y_gt},
            "id": sample.id,
            "gt": y_gt.eval_input,
        }


def train(
    model_client: adal.ModelClient,
    model_kwargs: Dict,
    train_batch_size=4,  # larger batch size is not that effective, probably because of llm's lost in the middle
    raw_shots: int = 36,
    bootstrap_shots: int = 4,
    max_steps=1,
    num_workers=4,
    strategy="constrained",
    optimization_order="sequential",
    debug=False,
    seed=None,
    tg: bool = False,
    max_proposals_per_step: int = 5,
    disable_backward=False,
    disable_backward_gradients=False,
):
    # TODO: ensure the teacher prompt gets updated with the new model
    adal_component = TrecClassifierAdal(
        model_client=model_client,
        model_kwargs=model_kwargs,
        text_optimizer_model_config=gpt_4o_model,
        backward_engine_model_config=gpt_4o_model,
        teacher_model_config=gpt_4o_model,
    )
    backward_pass_setup = None
    if tg:
        backward_pass_setup = BackwardPassSetup(
            all_pred_at_once=False,
            compute_grad_for_errors_only=False,
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
        exclude_input_fields_from_bootstrap_demos=False,
        max_proposals_per_step=max_proposals_per_step,
        disable_backward=disable_backward,
        disable_backward_gradients=disable_backward_gradients,
        text_optimizers_config_kwargs={"max_past_history": 2},
    )
    trainer.set_random_seed(seed)
    print(trainer)

    train_dataset, val_dataset, test_dataset = load_datasets()
    ckpt, _ = trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        debug=debug,
        backward_pass_setup=backward_pass_setup,
        # resume_from_ckpt="/Users/liyin/.adalflow/ckpt/TrecClassifierAdal/constrained_max_steps_12_5d1bf_run_1.json",
    )
    return ckpt


if __name__ == "__main__":
    # TODO:
    #     Evaluating step(6): 0.7333 across 30 samples, Max potential: 0.7778:  83%|▊| 30/36 [00:08<00:01,
    # Optimizer revert: 0.7096774193548387 <= 0.7777777777777778
    import json

    import random

    random.seed(2025)
    # np.random.seed(2025)  # Set NumPy random seed

    # make the strategy configurable in the script
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--strategy", type=str, default="constrained")
    parser.add_argument("--use_tg", action="store_true")
    parser.add_argument("--max_proposals_per_step", type=int, default=5)
    parser.add_argument(
        "output_path", nargs="?", help="File path to save the checkpoint"
    )
    parser.add_argument("--disable_backward", action="store_true")
    parser.add_argument("--disable_backward_gradients", action="store_true")

    args = parser.parse_args()

    set_strategy = args.strategy
    set_output_path = args.output_path
    use_tg = args.use_tg
    max_proposals_per_step = args.max_proposals_per_step
    disable_backward = args.disable_backward
    disable_backward_gradients = args.disable_backward_gradients

    ckpt = train(
        **gpt_3_model,
        debug=False,
        max_steps=12,
        strategy=set_strategy,
        optimization_order="sequential",
        seed=2025,
        tg=use_tg,
        max_proposals_per_step=max_proposals_per_step,
        disable_backward=disable_backward,
        disable_backward_gradients=disable_backward_gradients,
    )  # val 0.694 -> 0.833, #test 0.8472 -> 0.833, adding more shots does not help

    if set_output_path:
        with open(set_output_path, "w") as f:
            json.dump({"ckpt": ckpt}, f)
        print(f"Checkpoint saved to {set_output_path}")
    else:
        print("No file path provided for saving the checkpoint.")
