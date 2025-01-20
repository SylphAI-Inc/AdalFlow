import dspy.evaluate
from dspy.teleprompt import BootstrapFewShot
import dspy
from dspy import Example

from benchmarks.BHH_object_count.dspy_count import ObjectCount

turbo = dspy.OpenAI(model="gpt-3.5-turbo-0125")

gpt_4 = dspy.OpenAI(model="gpt-4o")
dspy.configure(lm=turbo)


def validate_exact_match(example, pred, trace=None):
    if dspy.evaluate.answer_exact_match(example, pred):
        acc = 1
    else:
        acc = 0
    return acc


def load_datasets():
    from use_cases.question_answering.bbh.data import load_datasets

    trainset, valset, testset = load_datasets()
    # dspy requires us to package the dataset to Example objects and specify the inputs

    dspy_trainset, dspy_valset, dspy_testset = [], [], []
    for dataset in zip(
        [trainset, valset, testset], [dspy_trainset, dspy_valset, dspy_testset]
    ):
        for item in dataset[0]:
            example = Example(question=item.question, answer=item.answer)
            example = example.with_inputs("question")
            dataset[1].append(example)

    return dspy_trainset, dspy_valset, dspy_testset


def train(dspy_trainset=None):

    print("Training on", len(dspy_trainset), "samples", dspy_trainset[0])

    teleprompter = BootstrapFewShot(metric=validate_exact_match)
    compiled_count = teleprompter.compile(
        ObjectCount(), teacher=ObjectCount(), trainset=dspy_trainset
    )
    return compiled_count


def train_MIPROv2(trainset, valset, save_path, filename):

    import os
    from dspy.teleprompt import MIPROv2

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tp = MIPROv2(
        metric=validate_exact_match,
        prompt_model=gpt_4,
        task_model=turbo,
        num_candidates=30,
        init_temperature=1.0,
    )
    compiled_task = tp.compile(
        ObjectCount(),
        trainset=trainset,
        valset=valset,
        max_bootstrapped_demos=5,
        max_labeled_demos=2,
        num_batches=12,  # MINIBATCH_SIZE = 25,
        seed=2025,
        requires_permission_to_run=False,
    )
    compiled_task.save(os.path.join(save_path, filename))
    return compiled_task


def validate(dataset, compiled_count):
    from tqdm import tqdm

    acc_list = []
    for item in tqdm(dataset):
        pred = compiled_count(item.question)
        acc = validate_exact_match(item, pred)
        acc_list.append(acc)
    return sum(acc_list) / len(acc_list)


def train_and_validate():
    save_path = "benchmarks/BHH_object_count/models/dspy"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    import time
    import tqdm

    dspy_trainset, dspy_valset, dspy_testset = load_datasets()

    val_accs = []
    test_accs = []
    training_times = []

    num_runs = 4

    for i in tqdm.tqdm(range(num_runs)):
        start = time.time()
        output_file = f"compiled_count_{i}.json"

        compiled_count = train_MIPROv2(
            dspy_trainset, dspy_valset, save_path, output_file
        )
        val_acc = validate(dspy_valset, compiled_count)
        test_acc = validate(dspy_testset, compiled_count)

        val_accs.append(val_acc)
        test_accs.append(test_acc)

        training_times.append(time.time() - start)

    # compute the mean and standard deviation
    import numpy as np

    val_accs = np.array(val_accs)
    test_accs = np.array(test_accs)
    training_times = np.array(training_times)

    print("Validation accuracy:", val_accs.mean(), val_accs.std())
    print("Test accuracy:", test_accs.mean(), test_accs.std())

    print("Training time:", training_times.mean())


if __name__ == "__main__":
    from adalflow.utils import setup_env

    import os

    setup_env()

    save_path = "benchmarks/BHH_object_count/models/dspy"

    train_and_validate()
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    # example = GenerateAnswer(
    #     question="How many musical instruments do I have?", answer="5"
    # )
    # pred = GenerateAnswer(
    #     question="How many musical instruments do I have?", answer="5"
    # )
    # print(validate_exact_match(example, pred))
