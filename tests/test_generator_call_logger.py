import unittest
from unittest.mock import patch, mock_open
from core.generator import GeneratorOutput

from tracing import GeneratorCallLogger


class TestGeneratorCallLogger(unittest.TestCase):
    def setUp(self):
        # This mock will cover os.makedirs in all methods used during the tests
        patcher = patch("os.makedirs")
        self.addCleanup(patcher.stop)  # Ensure that patcher is stopped after test
        self.mock_makedirs = patcher.start()

        self.logger = GeneratorCallLogger(dir="./fake/dir/")

        self.sample_output = GeneratorOutput(data="test data", error_message=None)

    @patch("os.makedirs")
    def test_register_generator(self, mock_makedirs):
        self.logger.register_generator("test_gen")
        self.assertIn("test_gen", self.logger.generator_names_to_files)
        self.assertEqual(
            self.logger.generator_names_to_files["test_gen"],
            "./fake/dir/test_gen.jsonl",
        )

        # Test re-registering
        with self.assertLogs(level="WARNING") as log:
            self.logger.register_generator("test_gen")
            self.assertIn("already registered", log.output[0])

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_log_call(self, mock_json_dump, mock_open, mock_exists):
        self.logger.register_generator("test_gen")
        self.logger.log_call(
            "test_gen", {"model": "test"}, {"input": "test"}, self.sample_output
        )

        # Ensure file write operations
        # Assert that open was called with the correct parameters
        mock_open.assert_called_with(
            "./fake/dir/test_gen.jsonl", mode="at", encoding="utf-8"
        )

    @patch("os.path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"prompt_kwargs": {}, "model_kwargs": {}, "output": {}, "time_stamp": "2021-01-01T00:00:00"}\n',
    )
    def test_load(self, mock_open, mock_exists):
        self.logger.register_generator("test_gen")
        records = self.logger.load("test_gen")
        self.assertIsInstance(records, list)
        self.assertEqual(len(records), 1)

        # Test load unregistered generator
        with self.assertRaises(Exception):
            self.logger.load("unregistered_gen")


if __name__ == "__main__":
    unittest.main()
