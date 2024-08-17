import random
from adalflow.optim.trainer.trainer import Trainer
from adalflow.utils.data import Subset, DataLoader

from use_cases.question_answering.bhh_object_count.task import (
    ObjectCountTaskFewShot,
    ObjectCountTaskOriginal,
)
from use_cases.question_answering.bhh_object_count.prepare_trainer import (
    TGDWithEvalFnLoss,
)
from use_cases.question_answering.bhh_object_count.config import (
    gpt_3_model,
    gpt_4o_model,
)
from adalflow.datasets.big_bench_hard import BigBenchHard


def train_object_count_text_grad_v1(
    batch_size=6,
    max_steps=1,
    max_samples=2,
    num_workers=2,
    strategy="random",
    debug=False,
):

    trainer = Trainer(
        optimizer_type="text-grad",
        strategy=strategy,
        max_steps=max_steps,
        num_workers=num_workers,
        adaltask=TGDWithEvalFnLoss(
            ObjectCountTaskOriginal, gpt_3_model, gpt_4o_model, gpt_4o_model
        ),
        ckpt_path="object_count_text_grad_random",
    )

    root = "cache_datasets"
    train_dataset = BigBenchHard("BBH_object_counting", split="train", root=root)
    val_dataset = BigBenchHard("BBH_object_counting", split="val", root=root)
    test_dataset = BigBenchHard("BBH_object_counting", split="test", root=root)

    def subset_dataset(dataset, num_samples):
        num_samples = min(num_samples, len(dataset))
        random_subset_indices = random.sample(range(len(dataset)), num_samples)
        return Subset(dataset, random_subset_indices)

    train_dataset = subset_dataset(train_dataset, max_samples)
    val_dataset = subset_dataset(val_dataset, max_samples)
    test_dataset = subset_dataset(test_dataset, max_samples)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    trainer.fit(
        train_loader=train_dataloader,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        debug=debug,
    )


def train_object_count_few_shot(
    total_shots=5,
    raw_shots: int = 3,
    bootstrap_shots: int = 2,
    max_steps=1,
    max_samples=2,
    num_workers=2,
    strategy="random",
    debug=False,
    **kwargs
):
    trainer = Trainer(
        optimizer_type="text-grad",
        strategy="random",
        max_steps=max_steps,
        num_workers=num_workers,
        adaltask=TGDWithEvalFnLoss(
            ObjectCountTaskFewShot, gpt_3_model, gpt_4o_model, gpt_4o_model
        ),
        ckpt_path="object_count_text_grad_few_shot",
        save_traces=True,
        raw_shots=raw_shots,
        bootstrap_shots=bootstrap_shots,
    )

    root = "cache_datasets"
    train_dataset = BigBenchHard("BBH_object_counting", split="train", root=root)
    val_dataset = BigBenchHard("BBH_object_counting", split="val", root=root)
    test_dataset = BigBenchHard("BBH_object_counting", split="test", root=root)

    def subset_dataset(dataset, num_samples):
        num_samples = min(num_samples, len(dataset))
        random_subset_indices = random.sample(range(len(dataset)), num_samples)
        return Subset(dataset, random_subset_indices)

    train_dataset = subset_dataset(train_dataset, max_samples)
    val_dataset = subset_dataset(val_dataset, max_samples)
    test_dataset = subset_dataset(test_dataset, max_samples)

    train_dataloader = DataLoader(  # noqa: F841
        train_dataset, batch_size=1, shuffle=True
    )  # noqa: F841
    # TODO: make it take a dataset instead of data loader
    trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        debug=debug,
    )


if __name__ == "__main__":
    # train_object_count_text_grad_v1(
    #     batch_size=12,
    #     max_steps=10,  # large batch size helps
    #     max_samples=100,
    #     num_workers=4,
    #     strategy="constrained",
    #     debug=False,
    # )

    train_object_count_few_shot(
        total_shots=5,
        raw_shots=0,
        bootstrap_shots=2,
        max_steps=10,
        max_samples=20,
        num_workers=4,
        strategy="random",
        debug=False,
    )
