import unittest
from unittest.mock import patch

from adalflow.components.model_client import RuncrateClient
from adalflow.components.model_client.runcrate_client import (
    BASE_URL,
    RuncrateClient as DirectRuncrateClient,
)
from adalflow.core.types import ModelType


class TestRuncrateClient(unittest.TestCase):
    def setUp(self):
        self.client = DirectRuncrateClient(api_key="fake_api_key")

    def test_runcrate_init(self):
        self.assertEqual(BASE_URL, "https://api.runcrate.ai/v1")
        self.assertEqual(self.client.base_url, "https://api.runcrate.ai/v1")
        self.assertEqual(self.client._input_type, "text")
        self.assertEqual(self.client._env_api_key_name, "RUNCRATE_API_KEY")

    def test_runcrate_is_registered_in_model_client_package(self):
        with patch.dict("os.environ", {"RUNCRATE_API_KEY": "env_api_key"}):
            client = RuncrateClient()
        self.assertEqual(client.base_url, "https://api.runcrate.ai/v1")

    def test_runcrate_init_sync_client(self):
        self.client.init_sync_client()
        self.assertEqual(self.client.sync_client.api_key, "fake_api_key")
        self.assertEqual(
            str(self.client.sync_client.base_url), "https://api.runcrate.ai/v1/"
        )

    def test_runcrate_init_async_client(self):
        self.client.async_client = self.client.init_async_client()
        self.assertEqual(self.client.async_client.api_key, "fake_api_key")
        self.assertEqual(
            str(self.client.async_client.base_url), "https://api.runcrate.ai/v1/"
        )

    @patch("os.getenv")
    def test_runcrate_reads_api_key_from_env(self, mock_os_getenv):
        mock_os_getenv.side_effect = lambda key, default=None: {
            "RUNCRATE_API_KEY": "env_api_key"
        }.get(key, default)
        client = DirectRuncrateClient()
        client.init_sync_client()
        self.assertEqual(client.sync_client.api_key, "env_api_key")

    def test_runcrate_custom_base_url(self):
        client = DirectRuncrateClient(
            api_key="fake_api_key", base_url="http://localhost:8080/v1"
        )
        self.assertEqual(client.base_url, "http://localhost:8080/v1")

    def test_runcrate_convert_inputs_to_api_kwargs(self):
        api_kwargs = self.client.convert_inputs_to_api_kwargs(
            input="Hello",
            model_kwargs={"model": "deepseek-ai/DeepSeek-V4-Flash", "temperature": 0},
            model_type=ModelType.LLM,
        )
        self.assertEqual(api_kwargs["model"], "deepseek-ai/DeepSeek-V4-Flash")
        self.assertEqual(api_kwargs["temperature"], 0)
        self.assertEqual(api_kwargs["input"], "Hello")


if __name__ == "__main__":
    unittest.main()
