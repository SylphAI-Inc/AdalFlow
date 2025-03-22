import unittest
from unittest.mock import MagicMock

from adalflow.optim.trainer import Trainer, AdalComponent
from adalflow.utils.data import DataLoader


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.train_loader = DataLoader(dataset=[1, 2, 3, 4, 5], batch_size=2)

    def test_no_train_dataset(self):
        trainer = Trainer(
            adaltask=MagicMock(spec=AdalComponent),
        )
        with self.assertRaises(ValueError):
            trainer.fit(train_dataset=None)

    def test_no_val_dataset(self):
        trainer = Trainer(
            adaltask=MagicMock(spec=AdalComponent),
        )
        with self.assertRaises(ValueError):
            trainer.fit(train_dataset=[1, 2, 3, 4, 5], val_dataset=None)


if __name__ == "__main__":
    unittest.main()
