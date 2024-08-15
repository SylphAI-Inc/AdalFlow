import unittest
from unittest.mock import patch, AsyncMock, Mock

# use the openai for mocking standard data types
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion

from adalflow.core.types import ModelType, GeneratorOutput
from adalflow.components.model_client.groq_client import GroqAPIClient


def getenv_side_effect(key):
    # This dictionary can hold more keys and values as needed
    env_vars = {"GROQ_API_KEY": "fake_api_key"}
    return env_vars.get(key, None)  # Returns None if key is not found


# modified from test_openai_client.py
class TestGroqClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = GroqAPIClient(api_key="fake_api_key")
        self.mock_response = {
            "id": "cmpl-3Q8Z5J9Z1Z5z5",
            "created": 1635820005,
            "object": "chat.completion",
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "message": {
                        "content": "Hello, world!",
                        "role": "assistant",
                    },
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": CompletionUsage(
                completion_tokens=10, prompt_tokens=20, total_tokens=30
            ),
        }
        self.mock_response = ChatCompletion(**self.mock_response)
        self.api_kwargs = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
        }

    @patch("adalflow.components.model_client.groq_client.AsyncGroq")
    async def test_acall_llm(self, MockAsyncGroq):
        mock_async_client = AsyncMock()
        MockAsyncGroq.return_value = mock_async_client

        # Mock the response

        mock_async_client.chat.completions.create = AsyncMock(
            return_value=self.mock_response
        )

        # Call the _acall method

        result = await self.client.acall(
            api_kwargs=self.api_kwargs, model_type=ModelType.LLM
        )

        # Assertions
        MockAsyncGroq.assert_called_once()
        mock_async_client.chat.completions.create.assert_awaited_once_with(
            **self.api_kwargs
        )
        self.assertEqual(result, self.mock_response)

    @patch.object(GroqAPIClient, "init_sync_client")
    @patch("adalflow.components.model_client.groq_client.Groq")
    def test_call(self, MockGroq, mock_init_sync_client):
        mock_sync_client = Mock()
        MockGroq.return_value = mock_sync_client
        mock_init_sync_client.return_value = mock_sync_client

        # Mock the client's api: chat.completions.create
        mock_sync_client.chat.completions.create = Mock(return_value=self.mock_response)

        # Set the sync client
        self.client.sync_client = mock_sync_client

        # Call the call method
        result = self.client.call(api_kwargs=self.api_kwargs, model_type=ModelType.LLM)

        # Assertions
        mock_sync_client.chat.completions.create.assert_called_once_with(
            **self.api_kwargs
        )
        self.assertEqual(result, self.mock_response)

        # test parse_chat_completion
        output = self.client.parse_chat_completion(completion=self.mock_response)
        self.assertTrue(isinstance(output, GeneratorOutput))
        self.assertEqual(output.raw_response, "Hello, world!")
        self.assertEqual(output.usage.completion_tokens, 10)
        self.assertEqual(output.usage.prompt_tokens, 20)
        self.assertEqual(output.usage.total_tokens, 30)


if __name__ == "__main__":
    unittest.main()
