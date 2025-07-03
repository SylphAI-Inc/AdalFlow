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

            self.generator = Generator(model_client=mock_api_client)
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
        # Verify GeneratorOutput has expected attributes
        self.assertTrue(hasattr(output, "data"))
        self.assertTrue(hasattr(output, "raw_response"))
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
        self._clean_up()
        generator = Generator(model_client=self.mock_api_client)
        prompt_logger = GeneratorStateLogger(
            save_dir=self.save_dir,
            project_name=self.project_name,
            filename=self.filename,
        )
        prompt_logger.log_prompt(generator=generator, name="Test Generator")
        print(f"""prompt_logger._trace_map: {prompt_logger._trace_map}""")
        self.assertTrue("Test Generator" in prompt_logger._trace_map)

        # Update the prompt variable and value
        # preset_prompt_kwargs = {"input_str": "Hello, updated world!"}
        # generator = Generator(
        #     model_client=self.mock_api_client, prompt_kwargs=preset_prompt_kwargs
        # )

        # prompt_logger.log_prompt(generator=generator, name="Test Generator")

        # print(f"""preset_prompt_kwargs: {prompt_logger._trace_map["Test Generator"]}""")
        # self.assertEqual(
        #     prompt_logger._trace_map["Test Generator"][1].prompt_states[
        #         "prompt_kwargs"
        #     ]["input_str"],
        #     "Hello, updated world!",
        # )

        # update the template
        # template = "Hello, {{ input_str }}!"
        # generator = Generator(model_client=self.mock_api_client, template=template)
        # prompt_logger.log_prompt(generator=generator, name="Test Generator")
        # self.assertEqual(
        #     prompt_logger._trace_map["Test Generator"][2].prompt_states["template"],
        #     "Hello, {{ input_str }}!",
        # )
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
        generator = Generator(model_client=self.client, template=template)

        # Call the generator and get the output
        output = generator.call(prompt_kwargs=prompt_kwargs, model_kwargs=model_kwargs)

        self.assertIsInstance(output, GeneratorOutput)
        print(f"output groq: {output}")
        # Verify GeneratorOutput structure for Groq client
        self.assertTrue(hasattr(output, "data"))
        self.assertTrue(hasattr(output, "raw_response"))
        # self.assertEqual(output.data, "Generated text response")


class TestGeneratorIntegration(unittest.TestCase):
    """Test Generator integration with Agent and Runner workflows."""

    def setUp(self):
        # Mock ModelClient for integration tests
        with patch(
            "adalflow.core.model_client.ModelClient", spec=ModelClient
        ) as MockAPI:
            mock_api_client = Mock(ModelClient)
            MockAPI.return_value = mock_api_client
            mock_api_client.call.return_value = "Integration test response"
            mock_api_client.parse_chat_completion.return_value = (
                "Integration test response"
            )
            self.mock_api_client = mock_api_client

    def test_generator_output_for_agent_planner(self):
        """Test that Generator produces output suitable for Agent planner use."""
        from adalflow.components.output_parsers import JsonOutputParser
        from adalflow.core.types import Function

        # Create a generator with Function output parser (like Agent planner)
        output_parser = JsonOutputParser(
            data_class=Function,
            return_data_class=True,
            include_fields=["thought", "name", "kwargs"],
        )

        generator = Generator(
            model_client=self.mock_api_client, output_processors=output_parser
        )

        # Mock the model client to return a JSON-like response
        self.mock_api_client.call.return_value = '{"thought": "I need to search", "name": "search", "kwargs": {"query": "test"}}'

        output = generator.call(prompt_kwargs={"input_str": "test query"})

        # Verify output is GeneratorOutput
        self.assertIsInstance(output, GeneratorOutput)
        # Verify it can be used by Agent/Runner workflow
        self.assertTrue(hasattr(output, "data"))

    def test_generator_template_integration(self):
        """Test Generator with custom template like Agent uses."""
        template = (
            "System: You are a helpful assistant.\nUser: {{input_str}}\nAssistant:"
        )

        generator = Generator(model_client=self.mock_api_client, template=template)

        # Test that generator accepts template and can generate prompt
        prompt = generator.get_prompt(input_str="Hello world")
        self.assertIn("Hello world", prompt)
        self.assertIn("System: You are a helpful assistant", prompt)

        # Test generation works with template
        output = generator.call(prompt_kwargs={"input_str": "Hello world"})
        self.assertIsInstance(output, GeneratorOutput)

    def test_generator_async_capability(self):
        """Test Generator async methods that Runner.acall uses."""

        async def async_test():
            # Mock async call
            async def async_mock_call(*args, **kwargs):
                return "Async response"

            self.mock_api_client.acall = async_mock_call

            generator = Generator(model_client=self.mock_api_client)

            # Test async call
            output = await generator.acall(prompt_kwargs={"input_str": "async test"})
            self.assertIsInstance(output, GeneratorOutput)

        import asyncio

        asyncio.run(async_test())

    def test_generator_training_mode(self):
        """Test Generator training mode that Agent.is_training() uses."""
        generator = Generator(model_client=self.mock_api_client)

        # Initially not in training mode
        self.assertFalse(generator.training)

        # Set to training mode
        generator.training = True
        self.assertTrue(generator.training)

        # Can switch back
        generator.training = False
        self.assertFalse(generator.training)

    def test_generator_prompt_kwargs_persistence(self):
        """Test Generator maintains prompt_kwargs like Agent planner needs."""
        initial_prompt_kwargs = {
            "tools": "[tool1, tool2]",
            "output_format_str": "JSON format",
            "task_desc": "Agent task",
            "max_steps": 10,
            "step_history": [],
        }

        generator = Generator(
            model_client=self.mock_api_client, prompt_kwargs=initial_prompt_kwargs
        )

        # Verify prompt_kwargs are stored
        self.assertEqual(generator.prompt_kwargs, initial_prompt_kwargs)

        # Test that additional kwargs can be passed to call
        output = generator.call(prompt_kwargs={"input_str": "test", "current_step": 1})
        self.assertIsInstance(output, GeneratorOutput)


if __name__ == "__main__":
    unittest.main()
