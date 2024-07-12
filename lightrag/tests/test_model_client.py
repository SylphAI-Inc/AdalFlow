from typing import Any
import unittest
from unittest.mock import patch

from lightrag.components.model_client import OpenAIClient as OpenAIClientLazyImport


class TestLazyImportSubclassing(unittest.TestCase):
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
        from lightrag.components.model_client.openai_client import OpenAIClient

        class CorrectCustomizeOpenAIClient(OpenAIClient):
            def __init__(self):
                super().__init__()

            def parse_chat_completion(self, completion: Any) -> Any:
                """Parse the completion to a str."""
                print(f"completion: {completion}")
                return self.chat_completion_parser(completion)

        CorrectCustomizeOpenAIClient()


if __name__ == "__main__":
    unittest.main()
