from typing import Dict

import adalflow as adal

from benchmarks.hotpot_qa.config import load_datasets

from benchmarks.hotpot_qa.adal_exp.build_multi_hop_rag import MultiHopRAG
from benchmarks.hotpot_qa.adal_exp.adal_task import HotPotQAAdal
from use_cases.config import gpt_3_model, gpt_4o_model


from adalflow.core.generator import BackwardPassSetup


def train_diagnose(
    model_client: adal.ModelClient,
    model_kwargs: Dict,
) -> Dict:

    trainset, valset, testset = load_datasets()

    task = MultiHopRAG(
        model_client=model_client,
        model_kwargs=model_kwargs,
        passages_per_hop=2,
        max_hops=2,
    )

    adal_component = HotPotQAAdal(
        task=task,
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
    raw_shots: int = 2,
    bootstrap_shots: int = 2,
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
    disable_backward=False,
    disable_backward_gradients=False,
):
    task = MultiHopRAG(
        **gpt_3_model,
        passages_per_hop=2,
        max_hops=2,
    )
    adal_component = HotPotQAAdal(
        task=task,
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
        disable_backward=disable_backward,
        text_optimizers_config_kwargs={"max_past_history": 5},
        disable_backward_gradients=disable_backward_gradients,
    )
    trainer.set_random_seed(seed)
    print(trainer)

    train_dataset, val_dataset, test_dataset = load_datasets()
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

    # task = MultiHopRAGAdal(**gpt_3_model)
    # print(task)

    # train_diagnose(**gpt_3_model)

    ckpt = train(
        debug=False,
        max_steps=12,
        seed=2025,  # pass the numpy seed
        tg=use_tg,
        strategy=set_strategy,
        max_proposals_per_step=max_proposals_per_step,
        disable_backward=disable_backward,
        disable_backward_gradients=disable_backward_gradients,
        # resume_from_ckpt="/Users/liyin/.adalflow/ckpt/MultiHopRAGAdal/constrained_max_steps_12_fde51_run_1.json",
        # resume_from_ckpt="/Users/liyin/.adalflow/ckpt/ValinaRAGAdal/random_max_steps_12_7c091_run_1.json",
    )
    print(f"ckpt: {ckpt}")
    if set_output_path:
        with open(set_output_path, "w") as f:
            json.dump({"ckpt": ckpt}, f)
        print(f"Checkpoint saved to {set_output_path}")
    else:
        print("No file path provided for saving the checkpoint.")
