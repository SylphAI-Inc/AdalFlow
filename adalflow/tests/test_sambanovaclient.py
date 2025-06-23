import unittest
from unittest.mock import patch, Mock, AsyncMock

from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion

from adalflow.components.model_client.sambanova_client import SambaNovaClient
from adalflow.core import Generator
from adalflow.core.types import ModelType, GeneratorOutput
from adalflow.utils import get_logger


def getenv_side_effect(key):
    """Mock environment variables for testing."""
    env_vars = {"SAMBANOVA_API_KEY": "fake_sambanova_api_key"}
    return env_vars.get(key, None)


class TestSambaNovaClient(unittest.IsolatedAsyncioTestCase):
    """Test SambaNova client with proper mocking."""

    def setUp(self):
        """Set up test fixtures."""
        self.log = get_logger(level="DEBUG")
        self.prompt_kwargs = {"input_str": "What is the meaning of life?"}

        # Mock response for testing
        self.mock_response = {
            "id": "cmpl-sambanova-test-123",
            "created": 1635820005,
            "object": "chat.completion",
            "model": "Meta-Llama-3.1-8B-Instruct",
            "choices": [
                {
                    "message": {
                        "content": "The meaning of life is to find purpose and meaning in our existence.",
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
        self.mock_response = ChatCompletion(**self.mock_response)

        self.api_kwargs = {
            "messages": [{"role": "user", "content": "What is the meaning of life?"}],
            "model": "Meta-Llama-3.1-8B-Instruct",
        }

    def test_sambanova_client_init(self):
        """Test SambaNova client initialization."""
        with patch("os.getenv", side_effect=getenv_side_effect):
            client = SambaNovaClient(api_key="fake_api_key")

            # Test basic properties
            self.assertEqual(client.base_url, "https://api.sambanova.ai/v1/")
            self.assertEqual(client._env_api_key_name, "SAMBANOVA_API_KEY")
            self.assertEqual(client._input_type, "text")

    @patch("os.getenv")
    def test_sambanova_init_sync_client(self, mock_os_getenv):
        """Test sync client initialization."""
        mock_os_getenv.return_value = "fake_api_key"
        client = SambaNovaClient(api_key="fake_api_key")

        # Test that sync client is properly initialized
        self.assertIsNotNone(client.sync_client)
        self.assertEqual(client.sync_client.api_key, "fake_api_key")
        self.assertEqual(client.sync_client.base_url, "https://api.sambanova.ai/v1/")

    @patch("os.getenv")
    def test_sambanova_init_async_client(self, mock_os_getenv):
        """Test async client initialization."""
        mock_os_getenv.return_value = "fake_api_key"
        client = SambaNovaClient(api_key="fake_api_key")

        # Initialize async client
        client.async_client = client.init_async_client()

        # Test that async client is properly initialized
        self.assertIsNotNone(client.async_client)
        self.assertEqual(client.async_client.api_key, "fake_api_key")
        self.assertEqual(client.async_client.base_url, "https://api.sambanova.ai/v1/")

    @patch("adalflow.components.model_client.openai_client.AsyncOpenAI")
    async def test_sambanova_acall_llm(self, MockAsyncOpenAI):
        """Test async LLM call."""
        with patch("os.getenv", side_effect=getenv_side_effect):
            client = SambaNovaClient(api_key="fake_api_key")

            mock_async_client = AsyncMock()
            MockAsyncOpenAI.return_value = mock_async_client
            mock_async_client.chat.completions.create = AsyncMock(
                return_value=self.mock_response
            )

            # Call the acall method
            result = await client.acall(
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
    def test_sambanova_call(self, MockOpenAI, mock_init_sync_client):
        """Test sync LLM call."""
        with patch("os.getenv", side_effect=getenv_side_effect):
            client = SambaNovaClient(api_key="fake_api_key")

            mock_sync_client = Mock()
            MockOpenAI.return_value = mock_sync_client
            mock_init_sync_client.return_value = mock_sync_client
            mock_sync_client.chat.completions.create = Mock(
                return_value=self.mock_response
            )

            # Set the sync client
            client.sync_client = mock_sync_client

            # Call the call method
            result = client.call(api_kwargs=self.api_kwargs, model_type=ModelType.LLM)

            # Assertions
            mock_sync_client.chat.completions.create.assert_called_once_with(
                **self.api_kwargs
            )
            self.assertEqual(result, self.mock_response)

            # Test parse_chat_completion
            output = client.parse_chat_completion(completion=self.mock_response)
            self.assertTrue(isinstance(output, GeneratorOutput))
            self.assertEqual(
                output.raw_response,
                "The meaning of life is to find purpose and meaning in our existence.",
            )
            self.assertEqual(output.usage.completion_tokens, 15)
            self.assertEqual(output.usage.prompt_tokens, 25)
            self.assertEqual(output.usage.total_tokens, 40)

    @patch(
        "adalflow.components.model_client.openai_client.OpenAIClient.init_sync_client"
    )
    @patch("adalflow.components.model_client.openai_client.OpenAI")
    def test_sambanova_generator_integration(self, MockOpenAI, mock_init_sync_client):
        """Test SambaNova client integration with Generator."""
        with patch("os.getenv", side_effect=getenv_side_effect):
            client = SambaNovaClient(api_key="fake_api_key")

            mock_sync_client = Mock()
            MockOpenAI.return_value = mock_sync_client
            mock_init_sync_client.return_value = mock_sync_client
            mock_sync_client.chat.completions.create = Mock(
                return_value=self.mock_response
            )

            # Set the sync client
            client.sync_client = mock_sync_client

            # Create generator with mocked client
            gen = Generator(
                model_client=client,
                model_kwargs={
                    "model": "Meta-Llama-3.1-8B-Instruct",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                },
            )

            # Test generator call - use __call__ method instead of generate
            response = gen(prompt_kwargs=self.prompt_kwargs)

            # Verify response
            self.assertIsNotNone(response)
            self.log.debug(f"Response: {response}")

            # Verify that the mock was called
            mock_sync_client.chat.completions.create.assert_called()

    def test_sambanova_convert_inputs_to_api_kwargs(self):
        """Test input conversion to API kwargs."""
        with patch("os.getenv", side_effect=getenv_side_effect):
            client = SambaNovaClient(api_key="fake_api_key")

            # Test text input conversion
            api_kwargs = client.convert_inputs_to_api_kwargs(
                input="Hello, world!",
                model_kwargs={
                    "model": "Meta-Llama-3.1-8B-Instruct",
                    "temperature": 0.7,
                },
                model_type=ModelType.LLM,
            )

            # Verify the structure
            self.assertIn("messages", api_kwargs)
            self.assertIn("model", api_kwargs)
            self.assertEqual(api_kwargs["model"], "Meta-Llama-3.1-8B-Instruct")
            self.assertEqual(api_kwargs["temperature"], 0.7)

            # Verify messages structure
            messages = api_kwargs["messages"]
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[0]["content"], "Hello, world!")

    def test_sambanova_from_dict_to_dict(self):
        """Test serialization and deserialization."""
        with patch("os.getenv", side_effect=getenv_side_effect):
            test_api_key = "fake_api_key"
            client = SambaNovaClient(api_key=test_api_key)

            # Test to_dict
            client_dict = client.to_dict()
            self.assertIn("data", client_dict)
            self.assertIn("_api_key", client_dict["data"])
            self.assertEqual(client_dict["data"]["_api_key"], test_api_key)

            # Test from_dict
            new_client = SambaNovaClient.from_dict(client_dict)
            self.assertEqual(new_client.to_dict(), client_dict)


if __name__ == "__main__":
    unittest.main()
