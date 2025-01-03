dspy_save_path = "benchmarks/BHH_object_count/models/dspy"
adal_save_path = "benchmarks/BHH_object_count/models/adal"

from adalflow.datasets.hotpot_qa import HotPotQA


def load_datasets():

    trainset = HotPotQA(split="train", size=20)  # 20
    valset = HotPotQA(split="val", size=50)  # 50
    testset = HotPotQA(split="test", size=50)  # to keep the same as the dspy #50
    print(f"trainset, valset: {len(trainset)}, {len(valset)}, example: {trainset[0]}")
    return trainset, valset, testset
