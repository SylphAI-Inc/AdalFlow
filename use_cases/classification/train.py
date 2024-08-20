from typing import Any, Callable, Dict, Tuple
import adalflow as adal
from use_cases.classification.trec_task_structured_output import (
    TRECClassifierStructuredOutput,
)

from use_cases.classification.data import load_datasets, TRECExtendedData

from adalflow.eval.answer_match_acc import AnswerMatchAcc
from use_cases.question_answering.bhh_object_count.config import (
    gpt_3_model,
    gpt_4o_model,
)


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
            eval_fn_desc="exact_match: 1 if str(y) == str(y_gt) else 0",
        )
        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            backward_engine_model_config=backward_engine_model_config,
            text_optimizer_model_config=text_optimizer_model_config,
            teacher_model_config=teacher_model_config,
        )

    def handle_one_task_sample(self, sample: TRECExtendedData):
        return self.task.call, {"question": sample.question, "id": sample.id}

    def evaluate_one_sample(
        self, sample: TRECExtendedData, y_pred: adal.GeneratorOutput
    ) -> float:
        y_label = -1
        if y_pred and y_pred.data is not None and y_pred.data.class_name is not None:
            y_label = y_pred.data.class_name
        return self.eval_fn(y_label, sample.class_name)

    def handle_one_loss_sample(
        self, sample: TRECExtendedData, y_pred: adal.Parameter, *args, **kwargs
    ) -> Tuple[Callable[..., Any], Dict]:
        # prepare for evaluation
        full_response = y_pred.full_response
        y_label = -1
        if (
            full_response
            and full_response.data is not None
            and full_response.data.class_name is not None
        ):
            y_label = full_response.data.class_name
            # y_label = int(full_response.data.class_index)

        y_pred.eval_input = y_label
        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.class_name,
            # eval_input=sample.class_index,
            eval_input=sample.class_name,
            requires_opt=False,
        )
        # print(f"y_label: {y_label}, y_gt_label: {sample.class_index}")
        return self.loss_fn, {"kwargs": {"y": y_pred, "y_gt": y_gt}}

    def configure_teacher_generator(self):
        super().configure_teacher_generator_helper(**self.teacher_model_config)

    def configure_backward_engine(self):
        super().configure_backward_engine_helper(**self.backward_engine_model_config)

    def configure_optimizers(self):
        to = super().configure_text_optimizer_helper(**self.text_optimizer_model_config)
        do = super().configure_demo_optimizer_helper()
        return to + do


def train(
    model_client: adal.ModelClient,
    model_kwargs: Dict,
    train_batch_size=4,  # larger batch size is not that effective, probably because of llm's lost in the middle
    raw_shots: int = 0,
    bootstrap_shots: int = 1,
    max_steps=1,
    num_workers=4,
    strategy="random",
    optimization_order="mix",
    debug=False,
):
    # TODO: ensure the teacher prompt gets updated with the new model
    adal_component = TrecClassifierAdal(
        model_client=model_client,
        model_kwargs=model_kwargs,
        text_optimizer_model_config=gpt_4o_model,
        backward_engine_model_config=gpt_4o_model,
        teacher_model_config=gpt_4o_model,
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
        exclude_input_fields_from_bootstrap_demos=True,
    )
    print(trainer)

    train_dataset, val_dataset, test_dataset = load_datasets()
    trainer.fit(
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        # val_dataset=val_dataset,
        # test_dataset=test_dataset,
        debug=debug,
        # resume_from_ckpt="/Users/liyin/.adalflow/ckpt/TrecClassifierAdal/constrained_max_steps_1_33c1a_run_1.json",
    )


if __name__ == "__main__":
    # TODO:
    #     Evaluating step(6): 0.7333 across 30 samples, Max potential: 0.7778:  83%|▊| 30/36 [00:08<00:01,
    # Optimizer revert: 0.7096774193548387 <= 0.7777777777777778
    train(
        **gpt_3_model,
        debug=False,
        max_steps=12,
        strategy="constrained",
        optimization_order="sequential"
    )
    # val 0.694 -> 0.833, #test 0.8472 -> 0.833, adding more shots does not help
    # NOTE: raw: 40, bootstrap: 4, max_steps: 8, strategy: random, val: 86.1, test: 86.8 (+4.2% compared with dspy)
    # NOTE: train task without output format: val: 0.67->0.805, test: 0.805-> 0.896 # best performing model (zero-shot)
    # NOTE: train with without output format, use new class_name: constrained_max_steps_12_bac8d_run_1.json
    # val: 0.77.8, test: 0.86.8 #constrained_max_steps_12_138d9_run_1.json

    # REsume from the above, continue another 12 steps: val: 77.78% tets: 86.81%
    # result from the above, use bootstrap 1 shot: test -> 88.19% #constrained_max_steps_12_2ffa7_run_4.json (with input)
    # result from the above, use bootstrap 1 shot: no improvement, 86.81% #constrained_max_steps_12_2ffa7_run_5.json (with only rational and answers)
    # result from above, use bootstrap 2 shots: use input:no improvement
    # bootstrap is not helpful
    # 40 shots, 1 bootstrap, continue from last best, 86.1 val, 90.28% tes
    # 40 shots, resume, no improvment
    # continue from last best, 3 bootstrap, 83.3 val, 86.1 test (only rational)
    # continue from last best, 3 bootstrap, (both input and rational)86.1 val, 82.64 test (not really better)
    # NOTE:
    # continue from last best, 1 bootstrap, (both input and rational)86.1 val, 86.1 test (not really better)
    # TrecClassifierAdal/constrained_max_steps_12_2ffa7_run_2.json


# theory: all few-shots demo or instruction, all so that the llm can reason better. Once it reches to its limits, no more shots can help or further instruction can.
# there might be a saturation point!!!
