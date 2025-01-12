import unittest
from unittest.mock import patch, Mock

# use the openai for mocking standard data types

from adalflow.core.types import ModelType, GeneratorOutput
from adalflow.components.model_client import BedrockAPIClient


def getenv_side_effect(key):
    # This dictionary can hold more keys and values as needed
    env_vars = {
        "AWS_ACCESS_KEY_ID": "fake_api_key",
        "AWS_SECRET_ACCESS_KEY": "fake_api_key",
        "AWS_REGION_NAME": "fake_api_key",
    }
    return env_vars.get(key, None)  # Returns None if key is not found


# modified from test_openai_client.py
class TestBedrockClient(unittest.TestCase):
    def setUp(self):
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

        self.api_kwargs = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
        }

    @patch.object(BedrockAPIClient, "init_sync_client")
    @patch("adalflow.components.model_client.bedrock_client.boto3")
    def test_call(self, MockBedrock, mock_init_sync_client):
        mock_sync_client = Mock()
        MockBedrock.return_value = mock_sync_client
        mock_init_sync_client.return_value = mock_sync_client

        # Mock the client's api: converse
        mock_sync_client.converse = Mock(return_value=self.mock_response)

        # Set the sync client
        self.client.sync_client = mock_sync_client

        # Call the call method
        result = self.client.call(api_kwargs=self.api_kwargs, model_type=ModelType.LLM)

        # Assertions
        mock_sync_client.converse.assert_called_once_with(**self.api_kwargs)
        self.assertEqual(result, self.mock_response)

        # test parse_chat_completion
        output = self.client.parse_chat_completion(completion=self.mock_response)
        self.assertTrue(isinstance(output, GeneratorOutput))
        self.assertEqual(output.raw_response, "Hello, world!")
        self.assertEqual(output.usage.prompt_tokens, 20)
        self.assertEqual(output.usage.completion_tokens, 10)
        self.assertEqual(output.usage.total_tokens, 30)


if __name__ == "__main__":
    unittest.main()
