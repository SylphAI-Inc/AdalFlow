import unittest
from unittest.mock import patch, AsyncMock, Mock
import os
import base64

from openai.types import CompletionUsage, Image
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk

from adalflow.core.types import ModelType, GeneratorOutput
from adalflow.components.model_client.openai_client import OpenAIClient
import asyncio


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
            "model": "gpt-4o",
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
        self.mock_vision_response = {
            "id": "cmpl-4Q8Z5J9Z1Z5z5",
            "created": 1635820005,
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {
                        "content": "The image shows a beautiful sunset over mountains.",
                        "role": "assistant",
                    },
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": CompletionUsage(
                completion_tokens=15, prompt_tokens=25, total_tokens=40
            ),
        }
        self.mock_vision_response = ChatCompletion(**self.mock_vision_response)
        self.mock_image_response = [
            Image(
                url="https://example.com/generated_image.jpg",
                b64_json=None,
                revised_prompt="A white siamese cat sitting elegantly",
                model="dall-e-3",
            )
        ]
        self.api_kwargs = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o",
        }
        self.vision_api_kwargs = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/image.jpg",
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ],
            "model": "gpt-4o",
        }
        self.image_generation_kwargs = {
            "model": "dall-e-3",
            "prompt": "a white siamese cat",
            "size": "1024x1024",
            "quality": "standard",
            "n": 1,
        }

        # Add streaming test data
        self.streaming_chunks = [
            ChatCompletionChunk(
                id="cmpl-123",
                object="chat.completion.chunk",
                created=1635820005,
                model="gpt-4",
                choices=[
                    {
                        "delta": {"content": "Once "},
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
            ),
            ChatCompletionChunk(
                id="cmpl-123",
                object="chat.completion.chunk",
                created=1635820005,
                model="gpt-4",
                choices=[
                    {
                        "delta": {"content": "upon "},
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
            ),
            ChatCompletionChunk(
                id="cmpl-123",
                object="chat.completion.chunk",
                created=1635820005,
                model="gpt-4",
                choices=[
                    {
                        "delta": {},
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
            ),
        ]

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
            "model": "gpt-4o",
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
            "model": "gpt-4o",
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

    @patch("adalflow.components.model_client.openai_client.AsyncOpenAI")
    async def test_acall_llm_with_vision(self, MockAsyncOpenAI):
        mock_async_client = AsyncMock()
        MockAsyncOpenAI.return_value = mock_async_client

        # Mock the vision model response
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=self.mock_vision_response
        )

        # Call the _acall method with vision model
        result = await self.client.acall(
            api_kwargs=self.vision_api_kwargs, model_type=ModelType.LLM
        )

        # Assertions
        MockAsyncOpenAI.assert_called_once()
        mock_async_client.chat.completions.create.assert_awaited_once_with(
            **self.vision_api_kwargs
        )
        self.assertEqual(result, self.mock_vision_response)

    @patch(
        "adalflow.components.model_client.openai_client.OpenAIClient.init_sync_client"
    )
    @patch("adalflow.components.model_client.openai_client.OpenAI")
    def test_call_with_vision(self, MockSyncOpenAI, mock_init_sync_client):
        mock_sync_client = Mock()
        MockSyncOpenAI.return_value = mock_sync_client
        mock_init_sync_client.return_value = mock_sync_client

        # Mock the vision model response
        mock_sync_client.chat.completions.create = Mock(
            return_value=self.mock_vision_response
        )

        # Set the sync client
        self.client.sync_client = mock_sync_client

        # Call the call method with vision model
        result = self.client.call(
            api_kwargs=self.vision_api_kwargs, model_type=ModelType.LLM
        )

        # Assertions
        mock_sync_client.chat.completions.create.assert_called_once_with(
            **self.vision_api_kwargs
        )
        self.assertEqual(result, self.mock_vision_response)

        # Test parse_chat_completion for vision model
        output = self.client.parse_chat_completion(completion=self.mock_vision_response)
        self.assertTrue(isinstance(output, GeneratorOutput))
        self.assertEqual(
            output.raw_response, "The image shows a beautiful sunset over mountains."
        )
        self.assertEqual(output.usage.completion_tokens, 15)
        self.assertEqual(output.usage.prompt_tokens, 25)
        self.assertEqual(output.usage.total_tokens, 40)

    def test_convert_inputs_to_api_kwargs_for_image_generation(self):
        # Test basic image generation
        result = self.client.convert_inputs_to_api_kwargs(
            input="a white siamese cat",
            model_kwargs={"model": "dall-e-3"},
            model_type=ModelType.IMAGE_GENERATION,
        )
        self.assertEqual(result["prompt"], "a white siamese cat")
        self.assertEqual(result["model"], "dall-e-3")
        self.assertEqual(result["size"], "1024x1024")  # default
        self.assertEqual(result["quality"], "standard")  # default
        self.assertEqual(result["n"], 1)  # default

        # Test image edit
        test_image = "test_image.jpg"
        test_mask = "test_mask.jpg"
        try:
            # Create test files
            with open(test_image, "wb") as f:
                f.write(b"fake image content")
            with open(test_mask, "wb") as f:
                f.write(b"fake mask content")

            result = self.client.convert_inputs_to_api_kwargs(
                input="a white siamese cat",
                model_kwargs={
                    "model": "dall-e-2",
                    "image": test_image,
                    "mask": test_mask,
                },
                model_type=ModelType.IMAGE_GENERATION,
            )
            self.assertEqual(result["prompt"], "a white siamese cat")
            self.assertEqual(result["model"], "dall-e-2")
            self.assertTrue(isinstance(result["image"], str))  # base64 encoded
            self.assertTrue(isinstance(result["mask"], str))  # base64 encoded
        finally:
            # Cleanup
            if os.path.exists(test_image):
                os.remove(test_image)
            if os.path.exists(test_mask):
                os.remove(test_mask)

    @patch("adalflow.components.model_client.openai_client.AsyncOpenAI")
    async def test_acall_image_generation(self, MockAsyncOpenAI):
        mock_async_client = AsyncMock()
        MockAsyncOpenAI.return_value = mock_async_client

        # Mock the image generation response
        mock_async_client.images.generate = AsyncMock(
            return_value=type("Response", (), {"data": self.mock_image_response})()
        )

        # Call the acall method with image generation
        result = await self.client.acall(
            api_kwargs=self.image_generation_kwargs,
            model_type=ModelType.IMAGE_GENERATION,
        )

        # Assertions
        MockAsyncOpenAI.assert_called_once()
        mock_async_client.images.generate.assert_awaited_once_with(
            **self.image_generation_kwargs
        )
        self.assertEqual(result, self.mock_image_response)

        # Test parse_image_generation_response
        output = self.client.parse_image_generation_response(result)
        self.assertTrue(isinstance(output, GeneratorOutput))
        self.assertEqual(output.data, "https://example.com/generated_image.jpg")

    @patch(
        "adalflow.components.model_client.openai_client.OpenAIClient.init_sync_client"
    )
    @patch("adalflow.components.model_client.openai_client.OpenAI")
    def test_call_image_generation(self, MockSyncOpenAI, mock_init_sync_client):
        mock_sync_client = Mock()
        MockSyncOpenAI.return_value = mock_sync_client
        mock_init_sync_client.return_value = mock_sync_client

        # Mock the image generation response
        mock_sync_client.images.generate = Mock(
            return_value=type("Response", (), {"data": self.mock_image_response})()
        )

        # Set the sync client
        self.client.sync_client = mock_sync_client

        # Call the call method with image generation
        result = self.client.call(
            api_kwargs=self.image_generation_kwargs,
            model_type=ModelType.IMAGE_GENERATION,
        )

        # Assertions
        mock_sync_client.images.generate.assert_called_once_with(
            **self.image_generation_kwargs
        )
        self.assertEqual(result, self.mock_image_response)

        # Test parse_image_generation_response
        output = self.client.parse_image_generation_response(result)
        self.assertTrue(isinstance(output, GeneratorOutput))
        self.assertEqual(output.data, "https://example.com/generated_image.jpg")

    def test_from_dict_to_dict(self):
        test_api_key = "fake_api"
        client = OpenAIClient(api_key=test_api_key)
        client_dict = client.to_dict()
        new_client = OpenAIClient.from_dict(client_dict)
        self.assertEqual(new_client.to_dict(), client_dict)

    @patch("adalflow.components.model_client.openai_client.OpenAI")
    def test_init_sync_client_with_headers_and_organization(self, MockOpenAI):
        headers = {"Custom-Header": "CustomValue"}
        organization = "test-organization"

        # First call happens during __init__
        client = OpenAIClient(
            api_key="fake_api_key",
            headers=headers,
            organization=organization,
        )

        # Clear previous calls so we only test the explicit one below
        MockOpenAI.reset_mock()

        # Now call init_sync_client explicitly to trigger the OpenAI call
        _ = client.init_sync_client()

        # Assert OpenAI was called with correct parameters
        MockOpenAI.assert_called_once_with(
            api_key="fake_api_key",
            base_url="https://api.openai.com/v1/",
            organization=organization,
            default_headers=headers,
        )

    @patch("adalflow.components.model_client.openai_client.AsyncOpenAI")
    async def test_init_async_client_with_headers_and_organization(
        self, MockAsyncOpenAI
    ):
        headers = {"Custom-Header": "CustomValue"}
        organization = "test-organization"

        # Manually assign an AsyncMock to the return value
        mock_async_client = AsyncMock()
        MockAsyncOpenAI.return_value = mock_async_client

        client = OpenAIClient(
            api_key="fake_api_key",
            headers=headers,
            organization=organization,
        )

        async_client = client.init_async_client()  # Do NOT await here

        MockAsyncOpenAI.assert_called_once_with(
            api_key="fake_api_key",
            base_url="https://api.openai.com/v1/",
            organization=organization,
            default_headers=headers,
        )
        self.assertEqual(async_client, mock_async_client)

    @patch("adalflow.components.model_client.openai_client.OpenAI")
    def test_call_with_custom_headers_and_organization(self, MockOpenAI):
        # Test that headers and organization are passed during a call
        headers = {"Custom-Header": "CustomValue"}
        organization = "test-organization"
        mock_sync_client = Mock()
        MockOpenAI.return_value = mock_sync_client

        client = OpenAIClient(
            api_key="fake_api_key",
            headers=headers,
            organization=organization,
        )
        client.sync_client = mock_sync_client

        # Mock the API call
        mock_sync_client.chat.completions.create = Mock(return_value=self.mock_response)

        # Call the method
        result = client.call(api_kwargs=self.api_kwargs, model_type=ModelType.LLM)

        # Assertions
        mock_sync_client.chat.completions.create.assert_called_once_with(
            **self.api_kwargs
        )
        self.assertEqual(result, self.mock_response)

    @patch("adalflow.components.model_client.openai_client.AsyncOpenAI")
    async def test_acall_with_custom_headers_and_organization(self, MockAsyncOpenAI):
        # Test that headers and organization are passed during an async call
        headers = {"Custom-Header": "CustomValue"}
        organization = "test-organization"
        mock_async_client = AsyncMock()
        MockAsyncOpenAI.return_value = mock_async_client

        client = OpenAIClient(
            api_key="fake_api_key",
            headers=headers,
            organization=organization,
        )
        client.async_client = mock_async_client

        # Mock the API call
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=self.mock_response
        )

        # Call the method
        result = await client.acall(
            api_kwargs=self.api_kwargs, model_type=ModelType.LLM
        )

        # Assertions
        mock_async_client.chat.completions.create.assert_awaited_once_with(
            **self.api_kwargs
        )
        self.assertEqual(result, self.mock_response)

    async def test_async_streaming(self):
        """Test the async streaming method for OpenAIClient."""
        # Setup mock
        mock_async_client = AsyncMock()

        # Create an async generator for the mock stream
        async def mock_stream():
            for chunk in self.streaming_chunks:
                yield chunk
                await asyncio.sleep(0.01)

        mock_async_client.chat.completions.create.return_value = mock_stream()
        self.client.async_client = mock_async_client

        # Test API kwargs for streaming
        api_kwargs = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a short story."},
            ],
            "stream": True,
            "max_tokens": 200,
        }

        # Call the async streaming method
        stream = await self.client.acall(api_kwargs, ModelType.LLM)

        # Verify the streaming parser is set
        self.assertEqual(
            self.client.chat_completion_parser,
            self.client.streaming_chat_completion_parser,
        )

        # Process the stream
        full_response = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content

        # Verify the response
        self.assertIn("Once upon", full_response)

        # Verify the API was called correctly
        mock_async_client.chat.completions.create.assert_called_once_with(**api_kwargs)

    async def test_parser_switching(self):
        """Test that parser switching works correctly."""
        # Initially should be non-streaming parser
        self.assertEqual(
            self.client.chat_completion_parser,
            self.client.non_streaming_chat_completion_parser,
        )

        # Setup mock for streaming call
        mock_async_client = AsyncMock()

        async def mock_stream():
            yield self.streaming_chunks[0]

        mock_async_client.chat.completions.create.return_value = mock_stream()
        self.client.async_client = mock_async_client

        # Test streaming call - should switch to streaming parser
        await self.client.acall(
            {"model": "gpt-4", "messages": [], "stream": True}, ModelType.LLM
        )
        self.assertEqual(
            self.client.chat_completion_parser,
            self.client.streaming_chat_completion_parser,
        )

        # Test non-streaming call - should switch back to non-streaming parser
        mock_async_client.chat.completions.create.return_value = self.mock_response
        await self.client.acall(
            {"model": "gpt-4", "messages": [], "stream": False}, ModelType.LLM
        )
        self.assertEqual(
            self.client.chat_completion_parser,
            self.client.non_streaming_chat_completion_parser,
        )


if __name__ == "__main__":
    unittest.main()
