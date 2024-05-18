import unittest
from typing import Any

import utils.setup_env

from unittest.mock import patch, AsyncMock

from core.data_classes import ModelType
from components.api_client import OpenAIClient


class TestOpenAIClient(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.client = OpenAIClient()

    @patch("components.api_client.AsyncOpenAI")
    @patch("components.api_client.os.getenv")
    async def test_acall_llm(self, mock_getenv: Any, MockAsyncOpenAI: Any):
        mock_getenv.return_value = "fake_api_key"
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
        # mock_getenv.assert_called_once_with("OPENAI_API_KEY")
        # MockAsyncOpenAI.assert_called_once()
        mock_async_client.chat.completions.create.assert_awaited_once_with(**api_kwargs)
        self.assertEqual(result, mock_response)

    @patch("components.api_client.AsyncOpenAI")
    @patch("components.api_client.os.getenv")
    async def test_acall_embedder(self, mock_getenv, MockAsyncOpenAI):
        mock_getenv.return_value = "fake_api_key"
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
        mock_getenv.assert_called_once_with("OPENAI_API_KEY")
        MockAsyncOpenAI.assert_called_once()
        mock_async_client.embeddings.create.assert_awaited_once_with(**api_kwargs)
        self.assertEqual(result, mock_response)


if __name__ == "__main__":
    unittest.main()
