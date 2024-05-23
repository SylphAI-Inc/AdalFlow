from typing import Dict, Sequence
from torchmetrics.classification import MulticlassF1Score
from torchmetrics import Accuracy
import torch
from torch import Tensor


class ClassifierEvaluator:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.macro_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def run(self, preds: Sequence[int], targets: Sequence[int]):
        # convert predict to tensor
        # 1 -> [0, 1,..., 0]
        preds_tensor = torch.zeros(len(preds), self.num_classes)
        preds_tensor[range(len(preds)), preds] = 1
        print(f"Preds tensor: {preds_tensor}")
        # convert target to tensor, which will only be the real int
        targets_tensor = torch.tensor(targets)
        print(f"Targets tensor: {targets_tensor}")
        macro_f1_score: Tensor = self.macro_f1(preds_tensor, targets_tensor)
        accuracy: Tensor = self.accuracy(preds_tensor, targets_tensor)
        return round(accuracy.item(), 3), round(macro_f1_score.item(), 3)


if __name__ == "__main__":
    evaluator = ClassifierEvaluator(num_classes=6)
    preds = [0, 1, 2, 2, 4, 5]
    targets = [0, 1, 2, 0, 0, 0]
    accuracy, macro_f1_score = evaluator.run(preds, targets)
    print(f"Accuracy: {accuracy}")
    print(f"Micro F1 Score: {macro_f1_score}")
    print(type(accuracy))
