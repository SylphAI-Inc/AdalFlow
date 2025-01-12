import unittest
from unittest.mock import Mock

# use the openai for mocking standard data types

from adalflow.core.types import ModelType, GeneratorOutput
from adalflow.components.model_client import BedrockAPIClient


class TestBedrockClient(unittest.TestCase):
    def setUp(self) -> None:
        self.client = BedrockAPIClient()
        self.mock_response = {
            "ResponseMetadata": {
                "RequestId": "43aec10a-9780-4bd5-abcc-857d12460569",
                "HTTPStatusCode": 200,
                "HTTPHeaders": {
                    "date": "Sat, 30 Nov 2024 14:27:44 GMT",
                    "content-type": "application/json",
                    "content-length": "273",
                    "connection": "keep-alive",
                    "x-amzn-requestid": "43aec10a-9780-4bd5-abcc-857d12460569",
                },
                "RetryAttempts": 0,
            },
            "output": {
                "message": {"role": "assistant", "content": [{"text": "Hello, world!"}]}
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 20, "outputTokens": 10, "totalTokens": 30},
            "metrics": {"latencyMs": 430},
        }
        self.mock_stream_response = {
            "ResponseMetadata": {
                "RequestId": "c76d625e-9fdb-4173-8138-debdd724fc56",
                "HTTPStatusCode": 200,
                "HTTPHeaders": {
                    "date": "Sun, 12 Jan 2025 15:10:00 GMT",
                    "content-type": "application/vnd.amazon.eventstream",
                    "transfer-encoding": "chunked",
                    "connection": "keep-alive",
                    "x-amzn-requestid": "c76d625e-9fdb-4173-8138-debdd724fc56",
                },
                "RetryAttempts": 0,
            },
            "stream": iter(()),
        }

        self.api_kwargs = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
        }

    def test_call(self) -> None:
        """Test that the converse method is called correctly."""
        mock_sync_client = Mock()

        # Mock the converse API calls.
        mock_sync_client.converse = Mock(return_value=self.mock_response)
        mock_sync_client.converse_stream = Mock(return_value=self.mock_response)

        # Set the sync client
        self.client.sync_client = mock_sync_client

        # Call the call method
        result = self.client.call(api_kwargs=self.api_kwargs, model_type=ModelType.LLM)

        # Assertions
        mock_sync_client.converse.assert_called_once_with(**self.api_kwargs)
        mock_sync_client.converse_stream.assert_not_called()
        self.assertEqual(result, self.mock_response)

        # test parse_chat_completion
        output = self.client.parse_chat_completion(completion=self.mock_response)
        self.assertTrue(isinstance(output, GeneratorOutput))
        self.assertEqual(output.raw_response, "Hello, world!")
        self.assertEqual(output.usage.prompt_tokens, 20)
        self.assertEqual(output.usage.completion_tokens, 10)
        self.assertEqual(output.usage.total_tokens, 30)

    def test_streaming_call(self) -> None:
        """Test that a streaming call calls the converse_stream method."""
        mock_sync_client = Mock()

        # Mock the converse API calls.
        mock_sync_client.converse_stream = Mock(return_value=self.mock_response)
        mock_sync_client.converse = Mock(return_value=self.mock_response)

        # Set the sync client.
        self.client.sync_client = mock_sync_client

        # Call the call method.
        stream_kwargs = self.api_kwargs | {"stream": True}
        self.client.call(api_kwargs=stream_kwargs, model_type=ModelType.LLM)

        # Assert the streaming call was made.
        mock_sync_client.converse_stream.assert_called_once_with(**stream_kwargs)
        mock_sync_client.converse.assert_not_called()

    def test_call_value_error(self) -> None:
        """Test that a ValueError is raised when an invalid model_type is passed."""
        mock_sync_client = Mock()

        # Set the sync client
        self.client.sync_client = mock_sync_client

        # Test that ValueError is raised
        with self.assertRaises(ValueError):
            self.client.call(
                api_kwargs={},
                model_type=ModelType.UNDEFINED,  # This should trigger ValueError
            )

    def test_parse_chat_completion(self) -> None:
        """Test that the parse_chat_completion does not call usage completion when
        streaming."""
        mock_track_completion_usage = Mock()
        self.client.track_completion_usage = mock_track_completion_usage

        self.client.chat_completion_parser = self.client.handle_stream_response
        generator_output = self.client.parse_chat_completion(self.mock_stream_response)

        mock_track_completion_usage.assert_not_called()
        assert isinstance(generator_output, GeneratorOutput)

    def test_parse_chat_completion_call_usage(self) -> None:
        """Test that the parse_chat_completion calls usage completion when streaming."""
        mock_track_completion_usage = Mock()
        self.client.track_completion_usage = mock_track_completion_usage
        generator_output = self.client.parse_chat_completion(self.mock_response)

        mock_track_completion_usage.assert_called_once()
        assert isinstance(generator_output, GeneratorOutput)


if __name__ == "__main__":
    unittest.main()
