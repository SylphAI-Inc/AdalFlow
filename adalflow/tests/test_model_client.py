from typing import Any
import unittest
from unittest.mock import patch

from adalflow.components.model_client import OpenAIClient as OpenAIClientLazyImport
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.core.types import GeneratorOutput


class TestLazyImportSubclassing(unittest.TestCase):
    def setUp(self):
        self.client = OpenAIClient(api_key="test")

    def test_subclassing_raises_error(self):
        with self.assertRaises(TypeError):

            class InvalidCustomizeOpenAIClient(OpenAIClientLazyImport):
                def __init__(self):
                    super().__init__()

                def parse_chat_completion(self, completion: Any) -> Any:
                    """Parse the completion to a str."""
                    print(f"completion: {completion}")
                    return self.chat_completion_parser(completion)

            InvalidCustomizeOpenAIClient()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_api_key"})
    def test_correct_subclassing(self):
        from adalflow.components.model_client.openai_client import OpenAIClient

        class CorrectCustomizeOpenAIClient(OpenAIClient):
            def __init__(self):
                super().__init__()

            def parse_chat_completion(self, completion: Any) -> Any:
                """Parse the completion to a str."""
                print(f"completion: {completion}")
                return self.chat_completion_parser(completion)

        CorrectCustomizeOpenAIClient()

    @patch.object(OpenAIClient, "parse_chat_completion")
    def test_parse_chat_completion(self, mock_parse_chat_completion):
        mock_parse_chat_completion.return_value = GeneratorOutput()
        result = self.client.parse_chat_completion(completion="completion")
        self.assertIsInstance(result, GeneratorOutput)
        mock_parse_chat_completion.assert_called_once_with(completion="completion")


if __name__ == "__main__":
    unittest.main()
