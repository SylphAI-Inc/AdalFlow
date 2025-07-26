import unittest
from unittest.mock import Mock
from dataclasses import dataclass

from adalflow.components.model_client.chat_completion_to_response_converter import (
    ChatCompletionToResponseConverter,
    StreamingState,
    SequenceNumber,
    FAKE_RESPONSES_ID,
)

# Mock OpenAI types for testing
try:
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    from openai.types.completion_usage import CompletionUsage
    from openai.types.responses import (
        Response,
        ResponseCreatedEvent,
        ResponseCompletedEvent,
        ResponseTextDeltaEvent,
        ResponseContentPartAddedEvent,
        ResponseContentPartDoneEvent,
        ResponseOutputItemAddedEvent,
        ResponseOutputItemDoneEvent,
        ResponseOutputMessage,
        ResponseOutputText,
        ResponseUsage,
    )
    from openai._streaming import Stream, AsyncStream
except ImportError:
    # Mock classes for testing if OpenAI is not available
    @dataclass
    class ChatCompletion:
        choices: list
        usage: object = None

    @dataclass
    class ChatCompletionChunk:
        choices: list
        usage: object = None

    @dataclass
    class CompletionUsage:
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0

    @dataclass
    class Response:
        id: str = ""
        created_at: float = 0.0
        model: str = ""
        object: str = ""
        output: list = None
        usage: object = None

    @dataclass
    class ResponseCreatedEvent:
        response: Response
        type: str
        sequence_number: int

    @dataclass
    class ResponseCompletedEvent:
        response: Response
        type: str
        sequence_number: int

    @dataclass
    class ResponseTextDeltaEvent:
        content_index: int
        delta: str
        item_id: str
        output_index: int
        type: str
        sequence_number: int

    @dataclass
    class ResponseContentPartAddedEvent:
        content_index: int
        item_id: str
        output_index: int
        part: object
        type: str
        sequence_number: int

    @dataclass
    class ResponseContentPartDoneEvent:
        content_index: int
        item_id: str
        output_index: int
        part: object
        type: str
        sequence_number: int

    @dataclass
    class ResponseOutputItemAddedEvent:
        item: object
        output_index: int
        type: str
        sequence_number: int

    @dataclass
    class ResponseOutputItemDoneEvent:
        item: object
        output_index: int
        type: str
        sequence_number: int

    @dataclass
    class ResponseOutputMessage:
        id: str
        content: list
        role: str
        type: str
        status: str

    @dataclass
    class ResponseOutputText:
        text: str
        type: str
        annotations: list

    @dataclass
    class ResponseUsage:
        input_tokens: int = 0
        output_tokens: int = 0
        total_tokens: int = 0

    class Stream:
        def __init__(self, data):
            self.data = data

    class AsyncStream:
        def __init__(self, data):
            self.data = data


class TestChatCompletionToResponseConverter(unittest.TestCase):
    """Test suite for ChatCompletionToResponseConverter"""

    def test_get_chat_completion_content_success(self):
        """Test successful content extraction from ChatCompletion"""
        mock_completion = Mock(spec=ChatCompletion)
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = Mock()
        mock_completion.choices[0].message.content = "Hello, world!"

        result = ChatCompletionToResponseConverter.get_chat_completion_content(
            mock_completion
        )
        self.assertEqual(result, "Hello, world!")

    def test_get_chat_completion_content_multiple_choices(self):
        """Test content extraction with multiple choices (should return first)"""
        mock_completion = Mock(spec=ChatCompletion)
        mock_completion.choices = [Mock(), Mock()]
        mock_completion.choices[0].message = Mock()
        mock_completion.choices[0].message.content = "First choice"
        mock_completion.choices[1].message = Mock()
        mock_completion.choices[1].message.content = "Second choice"

        result = ChatCompletionToResponseConverter.get_chat_completion_content(
            mock_completion
        )
        self.assertEqual(result, "First choice")

    def test_get_chat_completion_content_empty_content(self):
        """Test content extraction with empty content"""
        mock_completion = Mock(spec=ChatCompletion)
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = Mock()
        mock_completion.choices[0].message.content = ""

        result = ChatCompletionToResponseConverter.get_chat_completion_content(
            mock_completion
        )
        self.assertEqual(result, "")

    def test_get_chat_completion_content_none_content(self):
        """Test content extraction with None content"""
        mock_completion = Mock(spec=ChatCompletion)
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = Mock()
        mock_completion.choices[0].message.content = None

        result = ChatCompletionToResponseConverter.get_chat_completion_content(
            mock_completion
        )
        self.assertIsNone(result)


class TestStreamingState(unittest.TestCase):
    """Test suite for StreamingState"""

    def test_streaming_state_initialization(self):
        """Test StreamingState initialization"""
        state = StreamingState()
        self.assertFalse(state.started)
        self.assertIsNone(state.text_content_index_and_output)
        self.assertIsNone(state.refusal_content_index_and_output)
        self.assertEqual(state.function_calls, {})

    def test_streaming_state_started_flag(self):
        """Test StreamingState started flag"""
        state = StreamingState()
        state.started = True
        self.assertTrue(state.started)

    def test_streaming_state_text_content_assignment(self):
        """Test StreamingState text content assignment"""
        state = StreamingState()
        mock_output = Mock()
        state.text_content_index_and_output = (0, mock_output)

        self.assertEqual(state.text_content_index_and_output[0], 0)
        self.assertEqual(state.text_content_index_and_output[1], mock_output)

    def test_streaming_state_function_calls_assignment(self):
        """Test StreamingState function calls assignment"""
        state = StreamingState()
        mock_function_call = Mock()
        state.function_calls[0] = mock_function_call

        self.assertEqual(state.function_calls[0], mock_function_call)
        self.assertEqual(len(state.function_calls), 1)


class TestSequenceNumber(unittest.TestCase):
    """Test suite for SequenceNumber"""

    def test_sequence_number_initialization(self):
        """Test SequenceNumber initialization"""
        seq = SequenceNumber()
        self.assertEqual(seq._sequence_number, 0)

    def test_sequence_number_increment(self):
        """Test SequenceNumber increment functionality"""
        seq = SequenceNumber()

        # First call should return 0 and increment internal counter
        first = seq.get_and_increment()
        self.assertEqual(first, 0)

        # Second call should return 1 and increment internal counter
        second = seq.get_and_increment()
        self.assertEqual(second, 1)

        # Third call should return 2 and increment internal counter
        third = seq.get_and_increment()
        self.assertEqual(third, 2)

    def test_sequence_number_multiple_instances(self):
        """Test that multiple SequenceNumber instances are independent"""
        seq1 = SequenceNumber()
        seq2 = SequenceNumber()

        # Both should start at 0
        self.assertEqual(seq1.get_and_increment(), 0)
        self.assertEqual(seq2.get_and_increment(), 0)

        # They should increment independently
        self.assertEqual(seq1.get_and_increment(), 1)
        self.assertEqual(seq2.get_and_increment(), 1)
        self.assertEqual(seq1.get_and_increment(), 2)


class TestSyncHandleStream(unittest.TestCase):
    """Test suite for sync_handle_stream"""

    def test_sync_handle_stream_empty_stream(self):
        """Test sync_handle_stream with empty stream"""
        mock_response = Mock(spec=Response)
        mock_response.model_copy.return_value = mock_response
        mock_response.output = []
        mock_response.usage = None

        empty_stream = iter([])

        events = list(
            ChatCompletionToResponseConverter.sync_handle_stream(
                mock_response, empty_stream
            )
        )

        # Should have at least a response.completed event
        self.assertGreater(len(events), 0)
        # Last event should be response.completed
        self.assertEqual(events[-1].type, "response.completed")

    def test_sync_handle_stream_single_chunk_with_content(self):
        """Test sync_handle_stream with single chunk containing content"""
        mock_response = Mock(spec=Response)
        mock_response.model_copy.return_value = mock_response
        mock_response.output = []
        mock_response.usage = None

        # Create a mock chunk with content
        mock_chunk = Mock(spec=ChatCompletionChunk)
        mock_chunk.usage = None
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta = Mock()
        mock_chunk.choices[0].delta.content = "Hello"
        mock_chunk.choices[0].delta.refusal = None
        mock_chunk.choices[0].delta.tool_calls = None

        stream = iter([mock_chunk])

        events = list(
            ChatCompletionToResponseConverter.sync_handle_stream(mock_response, stream)
        )

        # Should have multiple events including response.created, text.delta, and response.completed
        self.assertGreater(len(events), 2)

        # First event should be response.created
        self.assertEqual(events[0].type, "response.created")

        # Should have text delta event
        text_delta_events = [
            e for e in events if e.type == "response.output_text.delta"
        ]
        self.assertGreater(len(text_delta_events), 0)
        self.assertEqual(text_delta_events[0].delta, "Hello")

        # Last event should be response.completed
        self.assertEqual(events[-1].type, "response.completed")

    def test_sync_handle_stream_multiple_chunks_with_content(self):
        """Test sync_handle_stream with multiple chunks containing content"""
        mock_response = Mock(spec=Response)
        mock_response.model_copy.return_value = mock_response
        mock_response.output = []
        mock_response.usage = None

        # Create multiple mock chunks
        chunks = []
        for i, content in enumerate(["Hello", " ", "world", "!"]):
            mock_chunk = Mock(spec=ChatCompletionChunk)
            mock_chunk.usage = None
            mock_chunk.choices = [Mock()]
            mock_chunk.choices[0].delta = Mock()
            mock_chunk.choices[0].delta.content = content
            mock_chunk.choices[0].delta.refusal = None
            mock_chunk.choices[0].delta.tool_calls = None
            chunks.append(mock_chunk)

        stream = iter(chunks)

        events = list(
            ChatCompletionToResponseConverter.sync_handle_stream(mock_response, stream)
        )

        # Should have multiple events
        self.assertGreater(len(events), 5)

        # First event should be response.created
        self.assertEqual(events[0].type, "response.created")

        # Should have multiple text delta events
        text_delta_events = [
            e for e in events if e.type == "response.output_text.delta"
        ]
        self.assertEqual(len(text_delta_events), 4)

        # Check that deltas are in order
        expected_deltas = ["Hello", " ", "world", "!"]
        for i, event in enumerate(text_delta_events):
            self.assertEqual(event.delta, expected_deltas[i])

        # Last event should be response.completed
        self.assertEqual(events[-1].type, "response.completed")

    def test_sync_handle_stream_with_usage(self):
        """Test sync_handle_stream with usage information"""
        mock_response = Mock(spec=Response)
        mock_response.model_copy.return_value = mock_response
        mock_response.output = []
        mock_response.usage = None

        # Create mock usage
        mock_usage = Mock(spec=CompletionUsage)
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_usage.prompt_tokens_details = None
        mock_usage.completion_tokens_details = None

        # Create a mock chunk with usage
        mock_chunk = Mock(spec=ChatCompletionChunk)
        mock_chunk.usage = mock_usage
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta = Mock()
        mock_chunk.choices[0].delta.content = "Hello"
        mock_chunk.choices[0].delta.refusal = None
        mock_chunk.choices[0].delta.tool_calls = None

        stream = iter([mock_chunk])

        events = list(
            ChatCompletionToResponseConverter.sync_handle_stream(mock_response, stream)
        )

        # Last event should be response.completed with usage
        completed_event = events[-1]
        self.assertEqual(completed_event.type, "response.completed")
        self.assertIsNotNone(completed_event.response.usage)
        self.assertEqual(completed_event.response.usage.input_tokens, 10)
        self.assertEqual(completed_event.response.usage.output_tokens, 5)
        self.assertEqual(completed_event.response.usage.total_tokens, 15)


class TestAsyncHandleStream(unittest.IsolatedAsyncioTestCase):
    """Test suite for async_handle_stream"""

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
        # Last event should be response.completed
        self.assertEqual(events[-1].type, "response.completed")

    async def test_async_handle_stream_single_chunk_with_content(self):
        """Test async_handle_stream with single chunk containing content"""
        mock_response = Mock(spec=Response)
        mock_response.model_copy.return_value = mock_response
        mock_response.output = []
        mock_response.usage = None

        # Create a mock chunk with content
        mock_chunk = Mock(spec=ChatCompletionChunk)
        mock_chunk.usage = None
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta = Mock()
        mock_chunk.choices[0].delta.content = "Hello"
        mock_chunk.choices[0].delta.refusal = None
        mock_chunk.choices[0].delta.tool_calls = None

        async def single_chunk_stream():
            yield mock_chunk

        events = []
        async for event in ChatCompletionToResponseConverter.async_handle_stream(
            mock_response, single_chunk_stream()
        ):
            events.append(event)

        # Should have multiple events including response.created, text.delta, and response.completed
        self.assertGreater(len(events), 2)

        # First event should be response.created
        self.assertEqual(events[0].type, "response.created")

        # Should have text delta event
        text_delta_events = [
            e for e in events if e.type == "response.output_text.delta"
        ]
        self.assertGreater(len(text_delta_events), 0)
        self.assertEqual(text_delta_events[0].delta, "Hello")

        # Last event should be response.completed
        self.assertEqual(events[-1].type, "response.completed")

    async def test_async_handle_stream_multiple_chunks_with_content(self):
        """Test async_handle_stream with multiple chunks containing content"""
        mock_response = Mock(spec=Response)
        mock_response.model_copy.return_value = mock_response
        mock_response.output = []
        mock_response.usage = None

        # Create multiple mock chunks
        chunks = []
        for i, content in enumerate(["Hello", " ", "world", "!"]):
            mock_chunk = Mock(spec=ChatCompletionChunk)
            mock_chunk.usage = None
            mock_chunk.choices = [Mock()]
            mock_chunk.choices[0].delta = Mock()
            mock_chunk.choices[0].delta.content = content
            mock_chunk.choices[0].delta.refusal = None
            mock_chunk.choices[0].delta.tool_calls = None
            chunks.append(mock_chunk)

        async def multi_chunk_stream():
            for chunk in chunks:
                yield chunk

        events = []
        async for event in ChatCompletionToResponseConverter.async_handle_stream(
            mock_response, multi_chunk_stream()
        ):
            events.append(event)

        # Should have multiple events
        self.assertGreater(len(events), 5)

        # First event should be response.created
        self.assertEqual(events[0].type, "response.created")

        # Should have multiple text delta events
        text_delta_events = [
            e for e in events if e.type == "response.output_text.delta"
        ]
        self.assertEqual(len(text_delta_events), 4)

        # Check that deltas are in order
        expected_deltas = ["Hello", " ", "world", "!"]
        for i, event in enumerate(text_delta_events):
            self.assertEqual(event.delta, expected_deltas[i])

        # Last event should be response.completed
        self.assertEqual(events[-1].type, "response.completed")


class TestConstants(unittest.TestCase):
    """Test suite for constants"""

    def test_fake_responses_id_constant(self):
        """Test FAKE_RESPONSES_ID constant"""
        self.assertEqual(FAKE_RESPONSES_ID, "fake_responses_id")
        self.assertIsInstance(FAKE_RESPONSES_ID, str)
        self.assertGreater(len(FAKE_RESPONSES_ID), 0)


if __name__ == "__main__":
    unittest.main()
