if __name__ == "__main__":
    from use_cases.question_answering.bhh_object_count.data import load_datasets
    from use_cases.question_answering.bhh_object_count.config import llama3_model
    from use_cases.question_answering.bhh_object_count.prepare_trainer import (
        TGDWithEvalFnLoss,
    )

    from lightrag.optim.trainer.trainer import Trainer

    trainset, valset, testset = load_datasets(max_samples=10)
    adaltask = TGDWithEvalFnLoss(
        task_model_config=llama3_model,
        backward_engine_model_config=llama3_model,
        optimizer_model_config=llama3_model,
    )

    trainer = Trainer(adaltask=adaltask)
    diagnose = trainer.diagnose(train_dataset=trainset)
    print(diagnose)

    # Diagnostic results run on trainer set, with all inputs and outputs tracked in ckpt/TGDWithEvalFnLoss/llm_counter_call
