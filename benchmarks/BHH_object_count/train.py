import dspy.evaluate
from dspy.teleprompt import BootstrapFewShot
import dspy
from dspy import Example

from benchmarks.BHH_object_count.dspy_count import ObjectCount


def validate_exact_match(example, pred, trace=None):
    if dspy.evaluate.answer_exact_match(example, pred):
        acc = 1
    else:
        acc = 0
    return acc


def load_datasets(max_samples=10):
    from use_cases.question_answering.bhh_object_count.data import load_datasets

    trainset, valset, testset = load_datasets(max_samples=max_samples)
    # dspy requires us to package the dataset to Example objects and specify the inputs

    dspy_trainset, dspy_valset, dspy_testset = [], [], []
    for dataset in zip(
        [trainset, valset, testset], [dspy_trainset, dspy_valset, dspy_testset]
    ):
        for item in dataset[0]:
            example = Example(question=item.x, answer=item.y)
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


def validate(dataset, compiled_count):
    from tqdm import tqdm

    acc_list = []
    for item in tqdm(dataset):
        pred = compiled_count(item.question)
        acc = validate_exact_match(item, pred)
        acc_list.append(acc)
    return sum(acc_list) / len(acc_list)


if __name__ == "__main__":
    from benchmarks.BHH_object_count.dspy_count import GenerateAnswer
    import os

    save_path = "benchmarks/BHH_object_count/models/dspy"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    example = GenerateAnswer(
        question="How many musical instruments do I have?", answer="5"
    )
    pred = GenerateAnswer(
        question="How many musical instruments do I have?", answer="5"
    )
    print(validate_exact_match(example, pred))

    dspy_trainset, dspy_valset, dspy_testset = load_datasets(max_samples=4)

    start_val_acc = validate(dspy_valset, ObjectCount())
    start_test_acc = validate(dspy_testset, ObjectCount())
    print("Starting validation accuracy:", start_val_acc)
    print("Starting test accuracy:", start_test_acc)
    pass

    compiled_count = train(dspy_trainset)
    val_acc = validate(dspy_valset, compiled_count)
    test_acc = validate(dspy_testset, compiled_count)
    compiled_count.save(os.path.join(save_path, "compiled_count.json"))
    print("Validation accuracy:", val_acc)
    print("Test accuracy:", test_acc)
