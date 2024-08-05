import random
from lightrag.optim.trainer.trainer import Trainer
from lightrag.utils.data import Subset, DataLoader

from use_cases.question_answering.bhh_object_count.prepare_trainer import (
    TGDWithEvalFnLoss,
)
from use_cases.question_answering.bhh_object_count.config import (
    gpt_3_model,
    gpt_4o_model,
)
from lightrag.datasets.big_bench_hard import BigBenchHard


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
        adaltask=TGDWithEvalFnLoss(gpt_3_model, gpt_4o_model, gpt_4o_model),
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


if __name__ == "__main__":
    train_object_count_text_grad_v1(
        batch_size=12,
        max_steps=10,  # large batch size helps
        max_samples=100,
        num_workers=4,
        strategy="constrained",
        debug=False,
    )
