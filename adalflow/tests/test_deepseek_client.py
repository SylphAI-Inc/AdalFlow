import unittest
from unittest.mock import patch

from adalflow.components.model_client.deepseek_client import DeepSeekClient


def getenv_side_effect(key):
    env_vars = {"DEEPSEEK_API_KEY": "fake_api_key"}
    return env_vars.get(key, None)


class TestDeepSeekClient(unittest.TestCase):
    def setUp(self):
        self.client = DeepSeekClient(api_key="fake_api_key")

    def test_deepseek_init(self):
        self.assertEqual(self.client.base_url, "https://api.deepseek.com/v1/")
        self.assertEqual(self.client._input_type, "messages")
        self.assertEqual(self.client._env_api_key_name, "DEEPSEEK_API_KEY")

    # mock os.getenv(self._env_api_key_name) with getenv_side_effect
    @patch("os.getenv")
    def test_deepseek_init_sync_client(self, mock_os_getenv):
        mock_os_getenv.return_value = "fake_api_key"
        self.client.init_sync_client()
        self.assertEqual(self.client.sync_client.api_key, "fake_api_key")
        self.assertEqual(
            self.client.sync_client.base_url, "https://api.deepseek.com/v1/"
        )

    @patch("os.getenv")
    def test_deepseek_init_async_client(self, mock_os_getenv):
        mock_os_getenv.return_value = "fake_api_key"
        self.client.async_client = self.client.init_async_client()
        self.assertEqual(self.client.async_client.api_key, "fake_api_key")
        self.assertEqual(
            self.client.async_client.base_url, "https://api.deepseek.com/v1/"
        )


if __name__ == "__main__":
    unittest.main()
