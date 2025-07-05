import unittest
from unittest.mock import Mock, patch
import pytest
from collections.abc import Iterator

from adalflow.components.model_client.anthropic_client import (
    AnthropicAPIClient,
    get_chat_completion_usage,
)
from adalflow.core.types import ModelType, GeneratorOutput, ResponseUsage
from adalflow.components.model_client.chat_completion_to_response_converter import (
    ChatCompletionToResponseConverter,
    StreamingState,
    SequenceNumber,
)

# Mock OpenAI types for testing
try:
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    from openai.types.completion_usage import CompletionUsage
    from openai.types.responses import Response
    from openai._streaming import Stream, AsyncStream
except ImportError:
    # Mock classes for testing if OpenAI is not available
    class ChatCompletion:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class ChatCompletionChunk:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class CompletionUsage:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class Response:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class Stream:
        def __init__(self, data):
            self.data = data

    class AsyncStream:
        def __init__(self, data):
            self.data = data


class TestAnthropicAPIClient(unittest.TestCase):
    """Test suite for AnthropicAPIClient"""

    def setUp(self):
        """Set up test fixtures"""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}):
            self.client = AnthropicAPIClient()

    def test_init_sync_client(self):
        """Test synchronous client initialization"""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}):
            client = AnthropicAPIClient()
            self.assertIsNotNone(client.sync_client)
            self.assertEqual(client.base_url, "https://api.anthropic.com/v1/")

    def test_init_async_client(self):
        """Test asynchronous client initialization"""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}):
            client = AnthropicAPIClient()
            async_client = client.init_async_client()
            self.assertIsNotNone(async_client)

    def test_missing_api_key(self):
        """Test that missing API key raises ValueError"""
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as context:
                AnthropicAPIClient()
            self.assertIn("ANTHROPIC_API_KEY", str(context.exception))

    def test_convert_inputs_to_api_kwargs_string_input(self):
        """Test converting string input to API kwargs"""
        result = self.client.convert_inputs_to_api_kwargs(
            input="Hello world",
            model_kwargs={"model": "claude-3-opus-20240229"},
            model_type=ModelType.LLM,
        )

        expected = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello world"}],
            "max_tokens": 4096,
        }
        self.assertEqual(result, expected)

    def test_convert_inputs_to_api_kwargs_list_input(self):
        """Test converting list input to API kwargs"""
        messages = [{"role": "user", "content": "Hello"}]
        result = self.client.convert_inputs_to_api_kwargs(
            input=messages,
            model_kwargs={"model": "claude-3-opus-20240229"},
            model_type=ModelType.LLM,
        )

        expected = {
            "model": "claude-3-opus-20240229",
            "messages": messages,
            "max_tokens": 4096,
        }
        self.assertEqual(result, expected)

    def test_convert_inputs_to_api_kwargs_unsupported_model_type(self):
        """Test that unsupported model type raises ValueError"""
        with self.assertRaises(ValueError) as context:
            self.client.convert_inputs_to_api_kwargs(
                input="Hello", model_type=ModelType.EMBEDDER
            )
        self.assertIn("not supported", str(context.exception))

    def test_parse_chat_completion_non_streaming(self):
        """Test parsing non-streaming ChatCompletion"""
        # Mock ChatCompletion
        mock_completion = Mock(spec=ChatCompletion)
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = Mock()
        mock_completion.choices[0].message.content = "Hello, world!"
        mock_completion.usage = Mock()
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 5
        mock_completion.usage.total_tokens = 15
        mock_completion.usage.prompt_tokens_details = None
        mock_completion.usage.completion_tokens_details = None

        result = self.client.parse_chat_completion(mock_completion)

        self.assertIsInstance(result, GeneratorOutput)
        self.assertEqual(result.raw_response, "Hello, world!")
        self.assertIsNone(result.error)
        self.assertEqual(result.api_response, mock_completion)

    def test_parse_chat_completion_streaming(self):
        """Test parsing streaming ChatCompletion"""
        # Mock stream
        mock_stream = Mock(spec=Iterator)

        result = self.client.parse_chat_completion(mock_stream)

        self.assertIsInstance(result, GeneratorOutput)
        self.assertIsNone(result.error)
        self.assertEqual(result.api_response, mock_stream)

    @patch(
        "adalflow.components.model_client.chat_completion_to_response_converter.ChatCompletionToResponseConverter.get_chat_completion_content"
    )
    def test_parse_chat_completion_error(self, mock_get_content):
        """Test error handling in parse_chat_completion"""
        # Mock completion
        mock_completion = Mock(spec=ChatCompletion)
        # Make the get_chat_completion_content method raise an exception
        mock_get_content.side_effect = Exception("Test error")

        result = self.client.parse_chat_completion(mock_completion)

        self.assertIsInstance(result, GeneratorOutput)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.raw_response, "")

    def test_track_completion_usage(self):
        """Test tracking completion usage"""
        mock_completion = Mock(spec=ChatCompletion)
        mock_completion.usage = Mock()
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 5
        mock_completion.usage.total_tokens = 15

        result = self.client.track_completion_usage(mock_completion)

        self.assertEqual(result.prompt_tokens, 10)
        self.assertEqual(result.completion_tokens, 5)
        self.assertEqual(result.total_tokens, 15)

    def test_track_completion_usage_no_usage(self):
        """Test tracking completion usage when no usage is available"""
        mock_completion = Mock(spec=ChatCompletion)
        mock_completion.usage = None

        result = self.client.track_completion_usage(mock_completion)

        self.assertEqual(result.prompt_tokens, 0)
        self.assertEqual(result.completion_tokens, 0)
        self.assertEqual(result.total_tokens, 0)

    def test_call_non_streaming(self):
        """Test synchronous non-streaming call"""
        # Setup mock
        mock_response = Mock(spec=ChatCompletion)
        self.client.sync_client.chat.completions.create = Mock(
            return_value=mock_response
        )

        # Test call
        api_kwargs = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = self.client.call(api_kwargs, ModelType.LLM)

        self.assertEqual(result, mock_response)
        self.client.sync_client.chat.completions.create.assert_called_once_with(
            **api_kwargs
        )

    def test_call_streaming(self):
        """Test synchronous streaming call"""
        # Setup mock
        mock_stream = Mock(spec=Stream)
        self.client.sync_client.chat.completions.create = Mock(return_value=mock_stream)

        # Test call
        api_kwargs = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        result = self.client.call(api_kwargs, ModelType.LLM)

        self.assertEqual(result, mock_stream)
        self.client.sync_client.chat.completions.create.assert_called_once_with(
            **api_kwargs
        )

    def test_call_unsupported_model_type(self):
        """Test that unsupported model type raises ValueError"""
        with self.assertRaises(ValueError) as context:
            self.client.call({}, ModelType.EMBEDDER)
        self.assertIn("not supported", str(context.exception))

    @pytest.mark.asyncio
    async def test_acall(self):
        """Test asynchronous call"""
        # Setup mock
        mock_response = Mock(spec=ChatCompletion)
        # Create a mock async client
        mock_async_client = Mock()
        mock_async_client.chat.completions.create = Mock(return_value=mock_response)
        self.client.async_client = mock_async_client

        # Test call
        api_kwargs = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = await self.client.acall(api_kwargs, ModelType.LLM)

        self.assertEqual(result, mock_response)
        mock_async_client.chat.completions.create.assert_called_once_with(**api_kwargs)

    @pytest.mark.asyncio
    async def test_acall_unsupported_model_type(self):
        """Test that unsupported model type raises ValueError in async call"""
        with self.assertRaises(ValueError) as context:
            await self.client.acall({}, ModelType.EMBEDDER)
        self.assertIn("not supported", str(context.exception))


class TestGetChatCompletionUsage(unittest.TestCase):
    """Test suite for get_chat_completion_usage function"""

    def test_get_chat_completion_usage_with_full_usage(self):
        """Test usage extraction with full usage information"""
        mock_completion = Mock(spec=ChatCompletion)
        mock_completion.usage = Mock()
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 5
        mock_completion.usage.total_tokens = 15

        # Mock detailed usage
        mock_completion.usage.prompt_tokens_details = Mock()
        mock_completion.usage.prompt_tokens_details.cached_tokens = 2
        mock_completion.usage.completion_tokens_details = Mock()
        mock_completion.usage.completion_tokens_details.reasoning_tokens = 3

        result = get_chat_completion_usage(mock_completion)

        self.assertIsInstance(result, ResponseUsage)
        self.assertEqual(result.input_tokens, 10)
        self.assertEqual(result.output_tokens, 5)
        self.assertEqual(result.total_tokens, 15)
        self.assertEqual(result.input_tokens_details.cached_tokens, 2)
        self.assertEqual(result.output_tokens_details.reasoning_tokens, 3)

    def test_get_chat_completion_usage_no_usage(self):
        """Test usage extraction when no usage is available"""
        mock_completion = Mock(spec=ChatCompletion)
        mock_completion.usage = None

        result = get_chat_completion_usage(mock_completion)

        self.assertIsInstance(result, ResponseUsage)
        self.assertEqual(result.input_tokens, 0)
        self.assertEqual(result.output_tokens, 0)
        self.assertEqual(result.total_tokens, 0)
        self.assertEqual(result.input_tokens_details.cached_tokens, 0)
        self.assertEqual(result.output_tokens_details.reasoning_tokens, 0)

    def test_get_chat_completion_usage_partial_details(self):
        """Test usage extraction with partial details"""
        mock_completion = Mock(spec=ChatCompletion)
        mock_completion.usage = Mock()
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 5
        mock_completion.usage.total_tokens = 15
        mock_completion.usage.prompt_tokens_details = None
        mock_completion.usage.completion_tokens_details = None

        result = get_chat_completion_usage(mock_completion)

        self.assertIsInstance(result, ResponseUsage)
        self.assertEqual(result.input_tokens, 10)
        self.assertEqual(result.output_tokens, 5)
        self.assertEqual(result.total_tokens, 15)
        self.assertEqual(result.input_tokens_details.cached_tokens, 0)
        self.assertEqual(result.output_tokens_details.reasoning_tokens, 0)


class TestChatCompletionToResponseConverter(unittest.TestCase):
    """Test suite for ChatCompletionToResponseConverter"""

    def test_get_chat_completion_content(self):
        """Test extracting content from ChatCompletion"""
        mock_completion = Mock(spec=ChatCompletion)
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = Mock()
        mock_completion.choices[0].message.content = "Test response"

        result = ChatCompletionToResponseConverter.get_chat_completion_content(
            mock_completion
        )
        self.assertEqual(result, "Test response")

    def test_streaming_state_initialization(self):
        """Test StreamingState initialization"""
        state = StreamingState()
        self.assertFalse(state.started)
        self.assertIsNone(state.text_content_index_and_output)
        self.assertIsNone(state.refusal_content_index_and_output)
        self.assertEqual(state.function_calls, {})

    def test_sequence_number_functionality(self):
        """Test SequenceNumber functionality"""
        seq = SequenceNumber()
        self.assertEqual(seq.get_and_increment(), 0)
        self.assertEqual(seq.get_and_increment(), 1)
        self.assertEqual(seq.get_and_increment(), 2)

    def test_sync_handle_stream_empty_stream(self):
        """Test sync_handle_stream with empty stream"""
        mock_response = Mock(spec=Response)
        mock_response.model_copy.return_value = mock_response
        mock_response.output = []
        mock_response.usage = None

        empty_stream = iter([])  # Empty stream

        events = list(
            ChatCompletionToResponseConverter.sync_handle_stream(
                mock_response, empty_stream
            )
        )

        # Should have at least a response.completed event
        self.assertGreater(len(events), 0)
        self.assertEqual(events[-1].type, "response.completed")

    @pytest.mark.asyncio
    async def test_async_handle_stream_empty_stream(self):
        """Test async_handle_stream with empty stream"""
        mock_response = Mock(spec=Response)
        mock_response.model_copy.return_value = mock_response
        mock_response.output = []
        mock_response.usage = None

        async def empty_async_stream():
            return
            yield  # Never reached

        events = []
        async for event in ChatCompletionToResponseConverter.async_handle_stream(
            mock_response, empty_async_stream()
        ):
            events.append(event)

        # Should have at least a response.completed event
        self.assertGreater(len(events), 0)
        self.assertEqual(events[-1].type, "response.completed")


if __name__ == "__main__":
    unittest.main()
