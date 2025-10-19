import unittest
from unittest.mock import patch, AsyncMock, Mock, MagicMock
import os
import base64

from openai.types import Image
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)

from adalflow.core.types import ModelType, GeneratorOutput
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.components.model_client.utils import extract_text_from_response_stream
import asyncio


def getenv_side_effect(key):
    # This dictionary can hold more keys and values as needed
    env_vars = {"OPENAI_API_KEY": "fake_api_key"}
    return env_vars.get(key, None)  # Returns None if key is not found


class TestOpenAIClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = OpenAIClient(api_key="fake_api_key")
        # Create a mock Response object with the required fields
        self.mock_response = Mock(spec=Response)
        self.mock_response.id = "resp-3Q8Z5J9Z1Z5z5"
        self.mock_response.created_at = 1635820005.0
        self.mock_response.model = "gpt-4o"
        self.mock_response.object = "response"
        self.mock_response.output_text = "Hello, world!"
        self.mock_response.usage = ResponseUsage(
            input_tokens=20,
            output_tokens=10,
            total_tokens=30,
            input_tokens_details={"cached_tokens": 0},
            output_tokens_details={"reasoning_tokens": 0},
        )
        self.mock_vision_response = Mock(spec=Response)
        self.mock_vision_response.id = "resp-4Q8Z5J9Z1Z5z5"
        self.mock_vision_response.created_at = 1635820005.0
        self.mock_vision_response.model = "gpt-4o"
        self.mock_vision_response.object = "response"
        self.mock_vision_response.output_text = (
            "The image shows a beautiful sunset over mountains."
        )
        self.mock_vision_response.usage = ResponseUsage(
            input_tokens=25,
            output_tokens=15,
            total_tokens=40,
            input_tokens_details={"cached_tokens": 0},
            output_tokens_details={"reasoning_tokens": 0},
        )
        self.mock_image_response = [
            Image(
                url="https://example.com/generated_image.jpg",
                b64_json=None,
                revised_prompt="A white siamese cat sitting elegantly",
                model="dall-e-3",
            )
        ]
        self.api_kwargs = {
            "input": "Hello",
            "model": "gpt-4o",
        }
        self.vision_api_kwargs = {
            "input": "Describe this image: https://example.com/image.jpg",
            "model": "gpt-4o",
        }
        self.image_generation_kwargs = {
            "model": "dall-e-3",
            "prompt": "a white siamese cat",
            "size": "1024x1024",
            "quality": "standard",
            "n": 1,
        }

        # Add streaming test data for response API using Mock objects
        mock_delta_event1 = Mock(spec=ResponseTextDeltaEvent)
        mock_delta_event1.type = "response.output_text.delta"
        mock_delta_event1.delta = "Once "

        mock_delta_event2 = Mock(spec=ResponseTextDeltaEvent)
        mock_delta_event2.type = "response.output_text.delta"
        mock_delta_event2.delta = "upon "

        mock_response_obj = Mock(spec=Response)
        mock_response_obj.id = "resp-123"
        mock_response_obj.created_at = 1635820005.0
        mock_response_obj.model = "gpt-4"
        mock_response_obj.object = "response"
        mock_response_obj.output_text = "Once upon "
        mock_response_obj.usage = ResponseUsage(
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
            input_tokens_details={"cached_tokens": 0},
            output_tokens_details={"reasoning_tokens": 0},
        )

        mock_completed_event = Mock(spec=ResponseCompletedEvent)
        mock_completed_event.type = "response.completed"
        mock_completed_event.response = mock_response_obj

        self.streaming_events = [
            mock_delta_event1,
            mock_delta_event2,
            mock_completed_event,
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
        # Test with single image URL - Response API uses message format for multimodal
        model_kwargs = {
            "model": "gpt-4o",
            "images": "https://example.com/image.jpg",
        }
        result = self.client.convert_inputs_to_api_kwargs(
            input="Describe this image",
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM,
        )
        print(result)
        # Response API expects message format with content array for multimodal
        expected_input = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image"},
                    {"type": "input_image", "image_url": "https://example.com/image.jpg"}
                ]
            }
        ]
        self.assertEqual(result["input"], expected_input)
        self.assertEqual(result["model"], "gpt-4o")

        # Test with multiple images - Response API uses message format for multimodal
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
        # Response API expects message format with content array for multimodal
        expected_input = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Compare these images"},
                    {"type": "input_image", "image_url": "https://example.com/image1.jpg"},
                    {"type": "input_image", "image_url": "https://example.com/image2.jpg"}
                ]
            }
        ]
        self.assertEqual(result["input"], expected_input)
        self.assertEqual(result["model"], "gpt-4o")

    @patch("adalflow.components.model_client.openai_client.AsyncOpenAI")
    async def test_acall_llm(self, MockAsyncOpenAI):
        mock_async_client = AsyncMock()
        MockAsyncOpenAI.return_value = mock_async_client

        # Mock the response

        mock_async_client.responses.create = AsyncMock(return_value=self.mock_response)

        # Call the _acall method

        result = await self.client.acall(
            api_kwargs=self.api_kwargs, model_type=ModelType.LLM
        )

        # Assertions
        MockAsyncOpenAI.assert_called_once()
        mock_async_client.responses.create.assert_awaited_once_with(**self.api_kwargs)
        self.assertEqual(result, self.mock_response)

    @patch(
        "adalflow.components.model_client.openai_client.OpenAIClient.init_sync_client"
    )
    @patch("adalflow.components.model_client.openai_client.OpenAI")
    def test_call(self, MockSyncOpenAI, mock_init_sync_client):
        mock_sync_client = Mock()
        MockSyncOpenAI.return_value = mock_sync_client
        mock_init_sync_client.return_value = mock_sync_client

        # Mock the client's api: responses.create
        mock_sync_client.responses.create = Mock(return_value=self.mock_response)

        # Set the sync client
        self.client.sync_client = mock_sync_client

        # Call the call method
        result = self.client.call(api_kwargs=self.api_kwargs, model_type=ModelType.LLM)

        # Assertions
        mock_sync_client.responses.create.assert_called_once_with(**self.api_kwargs)
        self.assertEqual(result, self.mock_response)

        # test parse_response
        output = self.client.parse_chat_completion(completion=self.mock_response)
        self.assertTrue(isinstance(output, GeneratorOutput))
        self.assertEqual(output.raw_response, "Hello, world!")
        self.assertEqual(output.usage.output_tokens, 10)
        self.assertEqual(output.usage.input_tokens, 20)
        self.assertEqual(output.usage.total_tokens, 30)

    @patch("adalflow.components.model_client.openai_client.AsyncOpenAI")
    async def test_acall_llm_with_vision(self, MockAsyncOpenAI):
        mock_async_client = AsyncMock()
        MockAsyncOpenAI.return_value = mock_async_client

        # Mock the vision model response
        mock_async_client.responses.create = AsyncMock(
            return_value=self.mock_vision_response
        )

        # Call the _acall method with vision model
        result = await self.client.acall(
            api_kwargs=self.vision_api_kwargs, model_type=ModelType.LLM
        )

        # Assertions
        MockAsyncOpenAI.assert_called_once()
        mock_async_client.responses.create.assert_awaited_once_with(
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
        mock_sync_client.responses.create = Mock(return_value=self.mock_vision_response)

        # Set the sync client
        self.client.sync_client = mock_sync_client

        # Call the call method with vision model
        result = self.client.call(
            api_kwargs=self.vision_api_kwargs, model_type=ModelType.LLM
        )

        # Assertions
        mock_sync_client.responses.create.assert_called_once_with(
            **self.vision_api_kwargs
        )
        self.assertEqual(result, self.mock_vision_response)

        # Test parse_response for vision model
        output = self.client.parse_chat_completion(completion=self.mock_vision_response)
        self.assertTrue(isinstance(output, GeneratorOutput))
        self.assertEqual(
            output.raw_response, "The image shows a beautiful sunset over mountains."
        )
        self.assertEqual(output.usage.output_tokens, 15)
        self.assertEqual(output.usage.input_tokens, 25)
        self.assertEqual(output.usage.total_tokens, 40)

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
        mock_sync_client.responses.create = Mock(return_value=self.mock_response)

        # Call the method
        result = client.call(api_kwargs=self.api_kwargs, model_type=ModelType.LLM)

        # Assertions
        mock_sync_client.responses.create.assert_called_once_with(**self.api_kwargs)
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
        mock_async_client.responses.create = AsyncMock(return_value=self.mock_response)

        # Call the method
        result = await client.acall(
            api_kwargs=self.api_kwargs, model_type=ModelType.LLM
        )

        # Assertions
        mock_async_client.responses.create.assert_awaited_once_with(**self.api_kwargs)
        self.assertEqual(result, self.mock_response)

    async def test_async_streaming(self):
        """Test the async streaming method for OpenAIClient."""
        # Setup mock
        mock_async_client = AsyncMock()

        # Create an async generator for the mock stream
        async def mock_stream():
            for event in self.streaming_events:
                yield event
                await asyncio.sleep(0.01)

        mock_async_client.responses.create.return_value = mock_stream()
        self.client.async_client = mock_async_client

        # Test API kwargs for streaming - Response API uses input as string
        api_kwargs = {
            "model": "gpt-4",
            "input": "You are a helpful assistant. Tell me a short story.",
            "stream": True,
            "max_tokens": 200,
        }

        # Call the async streaming method
        stream = await self.client.acall(api_kwargs, ModelType.LLM)

        # Verify the streaming parser is available (dynamic selection)
        self.assertIsNotNone(self.client.streaming_response_parser)

        # Process the stream
        full_response = ""
        async for event in stream:
            if hasattr(event, "delta"):  # Mock ResponseTextDeltaEvent
                full_response += event.delta
            elif hasattr(event, "response") and hasattr(
                event.response, "output_text"
            ):  # Mock ResponseCompletedEvent
                full_response = event.response.output_text

        # Verify the response
        self.assertIn("Once upon", full_response)

        # Verify the API was called correctly
        mock_async_client.responses.create.assert_called_once_with(**api_kwargs)

    async def test_parser_switching(self):
        """Test that parser switching works correctly."""
        # Initially should be non-streaming parser
        self.assertEqual(
            self.client.response_parser,
            self.client.non_streaming_response_parser,
        )

        # Setup mock for streaming call
        mock_async_client = AsyncMock()

        async def mock_stream():
            yield self.streaming_events[0]

        mock_async_client.responses.create.return_value = mock_stream()
        self.client.async_client = mock_async_client

        # Test streaming call - should switch to streaming parser
        await self.client.acall(
            {"model": "gpt-4", "input": "Hello", "stream": True}, ModelType.LLM
        )
        self.assertIsNotNone(self.client.streaming_response_parser)

        # Test non-streaming call - should switch back to non-streaming parser
        mock_async_client.responses.create.return_value = self.mock_response
        await self.client.acall(
            {"model": "gpt-4", "input": "Hello", "stream": False}, ModelType.LLM
        )
        self.assertIsNotNone(self.client.non_streaming_response_parser)

    def test_reasoning_model_response(self):
        """Test parsing of reasoning model responses with reasoning field."""
        # Create a mock Response with reasoning output
        mock_reasoning_response = Mock(spec=Response)
        mock_reasoning_response.id = "resp-reasoning-123"
        mock_reasoning_response.created_at = 1635820005.0
        mock_reasoning_response.model = "o1"
        mock_reasoning_response.object = "response"
        mock_reasoning_response.output_text = None  # Reasoning models may not have output_text
        
        # Mock output array with reasoning and message
        mock_reasoning_item = Mock()
        mock_reasoning_item.type = "reasoning"
        mock_reasoning_item.id = "rs_123"
        mock_reasoning_item.summary = [
            Mock(type="summary_text", text="I'm thinking about the problem step by step...")
        ]
        
        mock_message_item = Mock()
        mock_message_item.type = "message"
        mock_message_item.content = [
            Mock(type="output_text", text="The answer is 42.")
        ]
        
        mock_reasoning_response.output = [mock_reasoning_item, mock_message_item]
        mock_reasoning_response.usage = ResponseUsage(
            input_tokens=50,
            output_tokens=100,
            total_tokens=150,
            input_tokens_details={"cached_tokens": 0},
            output_tokens_details={"reasoning_tokens": 80},
        )
        
        # Parse the response
        result = self.client.parse_chat_completion(mock_reasoning_response)
        
        # Assertions
        self.assertIsInstance(result, GeneratorOutput)
        self.assertEqual(result.data, "The answer is 42.")
        self.assertIsNotNone(result.thinking)
        self.assertIn("thinking about the problem", result.thinking)
        # Check that usage was properly tracked
        self.assertEqual(result.usage.output_tokens, 100)
        self.assertEqual(result.usage.total_tokens, 150)

    def test_multimodal_input_with_images(self):
        """Test multimodal input with images (vision models)."""
        # Test with URL image
        url_kwargs = self.client.convert_inputs_to_api_kwargs(
            input="What's in this image?",
            model_kwargs={
                "model": "gpt-4o",
                "images": "https://example.com/image.jpg"
            },
            model_type=ModelType.LLM
        )
        
        # Should format as message with content array
        self.assertIn("input", url_kwargs)
        self.assertIsInstance(url_kwargs["input"], list)
        self.assertEqual(url_kwargs["input"][0]["role"], "user")
        content = url_kwargs["input"][0]["content"]
        self.assertIsInstance(content, list)
        
        # Check text content
        text_content = next((c for c in content if c["type"] == "input_text"), None)
        self.assertIsNotNone(text_content)
        self.assertEqual(text_content["text"], "What's in this image?")
        
        # Check image content
        image_content = next((c for c in content if c["type"] == "input_image"), None)
        self.assertIsNotNone(image_content)
        self.assertEqual(image_content["image_url"], "https://example.com/image.jpg")
        
        # Test with base64 image
        base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        base64_kwargs = self.client.convert_inputs_to_api_kwargs(
            input="Describe this image",
            model_kwargs={
                "model": "gpt-4o",
                "images": f"data:image/png;base64,{base64_image}"
            },
            model_type=ModelType.LLM
        )
        
        # Check base64 image content
        content = base64_kwargs["input"][0]["content"]
        image_content = next((c for c in content if c["type"] == "input_image"), None)
        self.assertIsNotNone(image_content)
        self.assertTrue(image_content["image_url"].startswith("data:image/png;base64,"))

    def test_image_generation_response(self):
        """Test parsing of image generation responses."""
        # Create a mock Response with image generation output
        mock_image_response = Mock(spec=Response)
        mock_image_response.id = "resp-img-gen-123"
        mock_image_response.created_at = 1635820005.0
        mock_image_response.model = "gpt-4o"
        mock_image_response.object = "response"
        mock_image_response.output_text = None
        
        # Mock output array with image generation call
        mock_image_item = Mock()
        mock_image_item.type = "image_generation_call"
        mock_image_item.result = "base64_encoded_image_data_here"
        
        mock_message_item = Mock()
        mock_message_item.type = "message"
        mock_message_item.content = [
            Mock(type="output_text", text="I've generated an image of a cat for you.")
        ]
        
        mock_image_response.output = [mock_image_item, mock_message_item]
        mock_image_response.usage = ResponseUsage(
            input_tokens=30,
            output_tokens=50,
            total_tokens=80,
            input_tokens_details={"cached_tokens": 0},
            output_tokens_details={"reasoning_tokens": 0},
        )
        
        # Parse the response
        result = self.client.parse_chat_completion(mock_image_response)
        
        # Assertions
        self.assertIsInstance(result, GeneratorOutput)
        self.assertEqual(result.data, "I've generated an image of a cat for you.")
        self.assertIsNotNone(result.images)
        self.assertEqual(result.images, ["base64_encoded_image_data_here"])


    def test_streaming_with_helper_function(self):
        """Test streaming response with text extraction helper."""
        # Create streaming events with proper structure
        event1 = Mock()
        event1.type = "response.created"
        
        event2 = Mock()
        event2.type = "response.output_text.delta"
        event2.delta = "Hello "
        
        event3 = Mock()
        event3.type = "response.output_text.delta"
        event3.delta = "world!"
        
        event4 = Mock()
        event4.type = "response.done"
        
        events = [event1, event2, event3, event4]
        
        # Test text extraction
        extracted_text = []
        for event in events:
            text = extract_text_from_response_stream(event)
            if text:
                extracted_text.append(text)
        
        # Assertions
        self.assertEqual(extracted_text, ["Hello ", "world!"])
        self.assertEqual("".join(extracted_text), "Hello world!")

    async def test_reasoning_model_streaming(self):
        """Test streaming with reasoning model responses."""
        # Setup mock
        mock_async_client = AsyncMock()
        
        # Create reasoning streaming events with proper structure
        async def mock_reasoning_stream():
            # Reasoning events
            event1 = Mock()
            event1.type = "reasoning.start"
            yield event1
            
            event2 = Mock()
            event2.type = "reasoning.delta"
            event2.delta = "Thinking..."
            yield event2
            
            # Text output events
            event3 = Mock()
            event3.type = "response.output_text.delta"
            event3.delta = "The answer "
            yield event3
            
            event4 = Mock()
            event4.type = "response.output_text.delta"
            event4.delta = "is 42."
            yield event4
            
            event5 = Mock()
            event5.type = "response.done"
            yield event5
        
        mock_async_client.responses.create.return_value = mock_reasoning_stream()
        self.client.async_client = mock_async_client
        
        # Call with reasoning model
        api_kwargs = {
            "model": "o1",
            "input": "What is the meaning of life?",
            "stream": True,
            "reasoning": {"effort": "medium", "summary": "auto"}
        }
        
        stream = await self.client.acall(api_kwargs, ModelType.LLM_REASONING)
        
        # Process the stream
        text_chunks = []
        async for event in stream:
            text = extract_text_from_response_stream(event)
            if text:
                text_chunks.append(text)
        
        # Assertions
        self.assertEqual("".join(text_chunks), "The answer is 42.")

    def test_multiple_images_input(self):
        """Test multimodal input with multiple images."""
        # Test with multiple images
        multi_image_kwargs = self.client.convert_inputs_to_api_kwargs(
            input="Compare these images",
            model_kwargs={
                "model": "gpt-4o",
                "images": [
                    "https://example.com/image1.jpg",
                    "https://example.com/image2.jpg",
                    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                ]
            },
            model_type=ModelType.LLM
        )
        
        # Check content array
        content = multi_image_kwargs["input"][0]["content"]
        
        # Should have 1 text + 3 images = 4 items
        self.assertEqual(len(content), 4)
        
        # Count image contents
        image_contents = [c for c in content if c["type"] == "input_image"]
        self.assertEqual(len(image_contents), 3)
        
        # Verify each image
        self.assertEqual(image_contents[0]["image_url"], "https://example.com/image1.jpg")
        self.assertEqual(image_contents[1]["image_url"], "https://example.com/image2.jpg")
        self.assertTrue(image_contents[2]["image_url"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
