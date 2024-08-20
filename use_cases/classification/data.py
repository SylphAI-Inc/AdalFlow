from adalflow.datasets.trec import TrecDataset
from adalflow.datasets.types import TrecData
from dataclasses import dataclass, field

_COARSE_LABELS = [
    "ABBR",
    "ENTY",
    "DESC",
    "HUM",
    "LOC",
    "NUM",
]


@dataclass
class TRECExtendedData(TrecData):
    rationale: str = field(
        metadata={
            "desc": "Your step-by-step reasoning to classify the question to class_name"
        },
        default=None,
    )
    __input_fields__ = ["question"]
    __output_fields__ = ["rationale", "class_name"]


def load_datasets():
    """Load the dataset"""
    train_data = TrecDataset(split="train")
    val_data = TrecDataset(split="val")
    test_data = TrecDataset(split="test")
    return train_data, val_data, test_data  # 0.694, 0.847
