from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch, Mock
import unittest
import os
import shutil
from pathlib import Path

from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion

from adalflow.core.types import GeneratorOutput
from adalflow.core.generator import Generator


from adalflow.core.model_client import ModelClient
from adalflow.components.model_client.groq_client import GroqAPIClient
from adalflow.tracing import GeneratorStateLogger
from adalflow.core.types import ModelType


class TestGenerator(IsolatedAsyncioTestCase):
    def setUp(self):
        # Assuming that OpenAIClient is correctly mocked and passed to Generator
        with patch(
            "adalflow.core.model_client.ModelClient", spec=ModelClient
        ) as MockAPI:
            mock_api_client = Mock(ModelClient)
            MockAPI.return_value = mock_api_client
            mock_api_client.call.return_value = "Generated text response"

            mock_api_client.parse_chat_completion.return_value = (
                "Generated text response"
            )
            self.mock_api_client = mock_api_client

            self.generator = Generator(model_client=mock_api_client, model_type=ModelType.LLM)
            self.save_dir = "./tests/log"
            self.project_name = "TestGenerator"
            self.filename = "prompt_logger_test.json"

    def _clean_up(self):
        dir_path = os.path.join(self.save_dir, self.project_name)

        # Use shutil.rmtree to remove the directory recursively
        shutil.rmtree(
            dir_path, ignore_errors=True
        )  # ignore_errors will prevent throwing an error if the directory doesn't exist

    def test_generator_call(self):
        prompt_kwargs = {"input_str": "Hello, world!"}
        model_kwargs = {"model": "gpt-3.5-turbo"}

        output = self.generator.call(
            prompt_kwargs=prompt_kwargs, model_kwargs=model_kwargs
        )
        self.assertIsInstance(output, GeneratorOutput)
        print(f"output: {output}")
        # self.assertEqual(output.data, "Generated text response")

    def test_cache_path(self):
        prompt_kwargs = {"input_str": "Hello, world!"}
        model_kwargs = {"model": "phi3.5:latest"}

        self.test_generator = Generator(
            model_client=self.mock_api_client,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
            use_cache=True,
        )

        # Convert the path to a string to avoid the TypeError
        cache_path = self.test_generator.get_cache_path()
        cache_path_str = str(cache_path)

        print(f"cache path: {cache_path}")

        # Check if the sanitized model string is in the cache path
        self.assertIn("phi3_5_latest", cache_path_str)

        # Check if the cache path exists as a file (or directory, depending on your use case)

        self.assertTrue(
            Path(cache_path).exists(), f"Cache path {cache_path_str} does not exist"
        )

    def test_generator_prompt_logger_first_record(self):
        # prompt_kwargs = {"input_str": "Hello, world!"}
        # model_kwargs = {"model": "gpt-3.5-turbo"}
        generator = Generator(model_client=self.mock_api_client)
        prompt_logger = GeneratorStateLogger(
            save_dir=self.save_dir,
            project_name=self.project_name,
            filename=self.filename,
        )
        prompt_logger.log_prompt(generator=generator, name="Test Generator")
        # Check if the prompt is logged
        self.assertTrue("Test Generator" in prompt_logger._trace_map)
        self._clean_up()

    def test_generator_prompt_update(self):
        generator = Generator(model_client=self.mock_api_client)
        prompt_logger = GeneratorStateLogger(
            save_dir=self.save_dir,
            project_name=self.project_name,
            filename=self.filename,
        )
        prompt_logger.log_prompt(generator=generator, name="Test Generator")
        self.assertTrue("Test Generator" in prompt_logger._trace_map)

        # Update the prompt variable and value
        preset_prompt_kwargs = {"input_str": "Hello, updated world!"}
        generator = Generator(
            model_client=self.mock_api_client, prompt_kwargs=preset_prompt_kwargs
        )

        prompt_logger.log_prompt(generator=generator, name="Test Generator")
        print(
            f"""preset_prompt_kwargs: {prompt_logger._trace_map["Test Generator"][-1].prompt_states}"""
        )
        self.assertEqual(
            prompt_logger._trace_map["Test Generator"][1].prompt_states["data"][
                "prompt_kwargs"
            ]["input_str"],
            "Hello, updated world!",
        )

        # update the template
        template = "Hello, {{ input_str }}!"
        generator = Generator(model_client=self.mock_api_client, template=template)
        prompt_logger.log_prompt(generator=generator, name="Test Generator")
        self.assertEqual(
            prompt_logger._trace_map["Test Generator"][2].prompt_states["data"][
                "template"
            ],
            "Hello, {{ input_str }}!",
        )
        self._clean_up()


def getenv_side_effect(key):
    # This dictionary can hold more keys and values as needed
    env_vars = {"GROQ_API_KEY": "fake_api_key"}
    return env_vars.get(key, None)  # Returns None if key is not found


class TestGeneratorWithGroqClient(unittest.TestCase):
    # @patch("os.getenv", side_effect=getenv_side_effect)
    def setUp(self) -> None:
        with patch(
            "os.getenv", side_effect=getenv_side_effect
        ):  # Mock the environment variable
            self.client = GroqAPIClient()
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

    @patch.object(GroqAPIClient, "call")
    def test_groq_client_call(self, mock_call):
        # Mock the response

        mock_call.return_value = self.mock_response

        # Define prompt and model kwargs
        prompt_kwargs = {"input_str": "Hello, world!"}
        model_kwargs = {"model": "gpt-3.5-turbo"}
        template = "Hello, {{ input_str }}!"

        # Initialize the Generator with the mocked client
        generator = Generator(model_client=self.client, template=template, model_type=ModelType.LLM)

        # Call the generator and get the output
        output = generator.call(prompt_kwargs=prompt_kwargs, model_kwargs=model_kwargs)

        self.assertIsInstance(output, GeneratorOutput)
        print(f"output groq: {output}")
        # self.assertEqual(output.data, "Generated text response")


if __name__ == "__main__":
    unittest.main()
