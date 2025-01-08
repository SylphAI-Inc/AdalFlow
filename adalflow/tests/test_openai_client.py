import unittest
from unittest.mock import patch, AsyncMock, Mock
import os
import base64

from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion

from adalflow.core.types import ModelType, GeneratorOutput
from adalflow.components.model_client.openai_client import OpenAIClient


def getenv_side_effect(key):
    # This dictionary can hold more keys and values as needed
    env_vars = {"OPENAI_API_KEY": "fake_api_key"}
    return env_vars.get(key, None)  # Returns None if key is not found


class TestOpenAIClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = OpenAIClient(api_key="fake_api_key")
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

    def test_encode_image(self):
        # Create a temporary test image file
        test_image_path = "test_image.jpg"
        test_content = b"fake image content"
        try:
            with open(test_image_path, "wb") as f:
                f.write(test_content)

            # Test successful encoding
            encoded = self.client._encode_image(test_image_path)
            self.assertEqual(encoded, base64.b64encode(test_content).decode("utf-8"))

            # Test file not found
            with self.assertRaises(ValueError) as context:
                self.client._encode_image("nonexistent.jpg")
            self.assertIn("Image file not found", str(context.exception))

        finally:
            # Cleanup
            if os.path.exists(test_image_path):
                os.remove(test_image_path)

    def test_prepare_image_content(self):
        # Test URL image
        url = "https://example.com/image.jpg"
        result = self.client._prepare_image_content(url)
        self.assertEqual(
            result,
            {"type": "image_url", "image_url": {"url": url, "detail": "auto"}},
        )

        # Test with custom detail level
        result = self.client._prepare_image_content(url, detail="high")
        self.assertEqual(
            result,
            {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
        )

        # Test with pre-formatted content
        pre_formatted = {
            "type": "image_url",
            "image_url": {"url": url, "detail": "low"},
        }
        result = self.client._prepare_image_content(pre_formatted)
        self.assertEqual(result, pre_formatted)

    def test_convert_inputs_to_api_kwargs_with_images(self):
        # Test with single image URL
        model_kwargs = {
            "model": "gpt-4-vision-preview",
            "images": "https://example.com/image.jpg",
        }
        result = self.client.convert_inputs_to_api_kwargs(
            input="Describe this image",
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM,
        )
        expected_content = [
            {"type": "text", "text": "Describe this image"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg", "detail": "auto"},
            },
        ]
        self.assertEqual(result["messages"][0]["content"], expected_content)

        # Test with multiple images
        model_kwargs = {
            "model": "gpt-4-vision-preview",
            "images": [
                "https://example.com/image1.jpg",
                "https://example.com/image2.jpg",
            ],
            "detail": "high",
        }
        result = self.client.convert_inputs_to_api_kwargs(
            input="Compare these images",
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM,
        )
        expected_content = [
            {"type": "text", "text": "Compare these images"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image1.jpg",
                    "detail": "high",
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image2.jpg",
                    "detail": "high",
                },
            },
        ]
        self.assertEqual(result["messages"][0]["content"], expected_content)

    @patch("adalflow.components.model_client.openai_client.AsyncOpenAI")
    async def test_acall_llm(self, MockAsyncOpenAI):
        mock_async_client = AsyncMock()
        MockAsyncOpenAI.return_value = mock_async_client

        # Mock the response

        mock_async_client.chat.completions.create = AsyncMock(
            return_value=self.mock_response
        )

        # Call the _acall method

        result = await self.client.acall(
            api_kwargs=self.api_kwargs, model_type=ModelType.LLM
        )

        # Assertions
        MockAsyncOpenAI.assert_called_once()
        mock_async_client.chat.completions.create.assert_awaited_once_with(
            **self.api_kwargs
        )
        self.assertEqual(result, self.mock_response)

    @patch(
        "adalflow.components.model_client.openai_client.OpenAIClient.init_sync_client"
    )
    @patch("adalflow.components.model_client.openai_client.OpenAI")
    def test_call(self, MockSyncOpenAI, mock_init_sync_client):
        mock_sync_client = Mock()
        MockSyncOpenAI.return_value = mock_sync_client
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
