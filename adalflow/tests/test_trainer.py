import unittest
from unittest.mock import patch, MagicMock

from adalflow.optim.trainer import Trainer, AdalComponent
from adalflow.utils.data import DataLoader


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.train_loader = DataLoader(dataset=[1, 2, 3, 4, 5], batch_size=2)
        self.trainer = Trainer(
            adaltask=MagicMock(spec=AdalComponent),
            train_loader=self.train_loader,
            optimizer_type="text-grad",
            strategy="random",
            max_steps=1000,
            num_workers=2,
            ckpt_path=None,
        )

    @patch("os.listdir")
    @patch("os.makedirs")
    @patch("os.getcwd", return_value="/fake/dir")
    def test_directory_creation(self, mock_getcwd, mock_makedirs, mock_listdir):
        # mock_getcwd.return_value = "/fake/dir"
        mock_listdir.return_value = []  # Simulate no existing checkpoint files
        hyperparms = self.trainer.gather_trainer_states()
        print(f"hyperparms: {hyperparms}")

        self.trainer.prep_ckpt_file_path(hyperparms)
        print(f"ckpt_path: {self.trainer.ckpt_path}")
        print(f"adaltask class: {self.trainer.adaltask.__class__.__name__}")

        expected_ckpt_path = "/fake/dir/ckpt/AdalComponent"
        self.assertEqual(self.trainer.ckpt_path, expected_ckpt_path)
        mock_makedirs.assert_called_once_with(expected_ckpt_path, exist_ok=True)

        # check file naming
        print(f"ckpt_file: {self.trainer.ckpt_file}")
        self.assertTrue(
            self.trainer.ckpt_file.startswith(expected_ckpt_path),
            "Checkpoint file path does not start with expected path",
        )
        self.assertTrue(
            "run_1.json" in self.trainer.ckpt_file,
            "Checkpoint file path does not end with expected filename",
        )


if __name__ == "__main__":
    unittest.main()
