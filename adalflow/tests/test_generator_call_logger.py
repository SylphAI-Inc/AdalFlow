import unittest
import os
from unittest.mock import patch, mock_open

from adalflow.core.generator import GeneratorOutput

from adalflow.tracing import GeneratorCallLogger


class TestGeneratorCallLogger(unittest.TestCase):
    def setUp(self):
        # # This mock will cover os.makedirs in all methods used during the tests
        # patcher = patch("os.makedirs")
        # self.addCleanup(patcher.stop)  # Ensure that patcher is stopped after test
        # self.mock_makedirs = patcher.start()

        # Patch 'os.makedirs' for all instance methods
        patcher_makedirs = patch("os.makedirs")
        self.mock_makedirs = patcher_makedirs.start()
        self.addCleanup(patcher_makedirs.stop)

        # Patch 'open' to avoid filesystem access
        patcher_open = patch("builtins.open", mock_open())
        self.mock_open = patcher_open.start()
        self.addCleanup(patcher_open.stop)

        self.save_dir = "./fake/dir/"
        self.project_name = "TestGeneratorCallLogger"

        self.logger = GeneratorCallLogger(
            save_dir=self.save_dir, project_name=self.project_name
        )

        self.sample_output = GeneratorOutput(data="test data", error=None)

    def test_register_generator(self):
        self.logger.register_generator("test_gen")
        self.assertIn("test_gen", self.logger.generator_names_to_files)
        self.assertEqual(
            self.logger.generator_names_to_files["test_gen"],
            os.path.join(self.save_dir, self.project_name, "test_gen_call.jsonl"),
        )

        # Test re-registering
        with self.assertLogs(level="WARNING") as log:
            self.logger.register_generator("test_gen")
            self.assertIn("already registered", log.output[0])

    @patch("json.dump")
    def test_log_call(self, mock_json_dump):
        self.logger.register_generator("test_gen")
        self.logger.log_call(
            "test_gen", {"model": "test"}, {"input": "test"}, self.sample_output
        )

        file = self.logger.generator_names_to_files["test_gen"]

        # Ensure file write operations
        # Assert that open was called with the correct parameters
        self.mock_open.assert_called_with(file, mode="at", encoding="utf-8")

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
