import unittest
from unittest.mock import patch, AsyncMock

from lightrag.core.data_classes import ModelType
from lightrag.components.api_client import OpenAIClient


def getenv_side_effect(key):
    # This dictionary can hold more keys and values as needed
    env_vars = {"OPENAI_API_KEY": "fake_api_key"}
    return env_vars.get(key, None)  # Returns None if key is not found


class TestOpenAIClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = OpenAIClient(api_key="fake_api_key")

    @patch("components.api_client.openai_client.AsyncOpenAI")
    async def test_acall_llm(self, MockAsyncOpenAI):
        mock_async_client = AsyncMock()
        MockAsyncOpenAI.return_value = mock_async_client

        # Mock the response
        mock_response = {"choices": [{"message": {"content": "Hello, world!"}}]}
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        # Call the _acall method
        api_kwargs = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
        }
        result = await self.client.acall(
            api_kwargs=api_kwargs, model_type=ModelType.LLM
        )

        # Assertions
        MockAsyncOpenAI.assert_called_once()
        mock_async_client.chat.completions.create.assert_awaited_once_with(**api_kwargs)
        self.assertEqual(result, mock_response)

    @patch("components.api_client.openai_client.AsyncOpenAI")
    async def test_acall_embedder(self, MockAsyncOpenAI):
        mock_async_client = AsyncMock()
        MockAsyncOpenAI.return_value = mock_async_client

        # Mock the response
        mock_response = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_async_client.embeddings.create = AsyncMock(return_value=mock_response)

        # Call the _acall method
        api_kwargs = {"input": ["Hello, world!"], "model": "text-embedding-3-small"}
        result = await self.client.acall(
            api_kwargs=api_kwargs, model_type=ModelType.EMBEDDER
        )

        # Assertions
        MockAsyncOpenAI.assert_called_once()
        mock_async_client.embeddings.create.assert_awaited_once_with(**api_kwargs)
        self.assertEqual(result, mock_response)


if __name__ == "__main__":
    unittest.main()
