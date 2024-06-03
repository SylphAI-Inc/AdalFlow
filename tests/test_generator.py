import pytest
import asyncio
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch, AsyncMock, Mock
import os

from core.data_classes import GeneratorOutput
from core.generator import Generator
from components.api_client import OpenAIClient
from tracing import GeneratorStatesLogger
import utils.setup_env


class TestGenerator(IsolatedAsyncioTestCase):
    def setUp(self):
        # Assuming that OpenAIClient is correctly mocked and passed to Generator
        with patch("components.api_client.OpenAIClient", spec=OpenAIClient) as MockAPI:
            mock_api_client = Mock(OpenAIClient)
            MockAPI.return_value = mock_api_client
            mock_api_client.call.return_value = "Generated text response"

            mock_api_client.parse_chat_completion.return_value = (
                "Generated text response"
            )
            self.mock_api_client = mock_api_client

            self.generator = Generator(model_client=mock_api_client)
            self.prompt_filename = "./tests/log/prompt_logger_test.json"

    def _clean_up(self):
        try:
            os.remove(self.prompt_filename)
            os.rmdir("./tests/log")
        except FileNotFoundError:
            pass

    def test_generator_call(self):
        prompt_kwargs = {"input_str": "Hello, world!"}
        model_kwargs = {"model": "gpt-3.5-turbo"}

        output = self.generator.call(
            prompt_kwargs=prompt_kwargs, model_kwargs=model_kwargs
        )
        self.assertIsInstance(output, GeneratorOutput)
        self.assertEqual(output.data, "Generated text response")

    def test_generator_prompt_logger_first_record(self):
        # prompt_kwargs = {"input_str": "Hello, world!"}
        # model_kwargs = {"model": "gpt-3.5-turbo"}
        generator = Generator(model_client=self.mock_api_client)
        prompt_logger = GeneratorStatesLogger(filename=self.prompt_filename)
        prompt_logger.log_prompt(generator=generator, name="Test Generator")
        # Check if the prompt is logged
        self.assertTrue("Test Generator" in prompt_logger._trace_map)
        self._clean_up()

    def test_generator_prompt_update(self):
        generator = Generator(model_client=self.mock_api_client)
        prompt_logger = GeneratorStatesLogger(filename=self.prompt_filename)
        prompt_logger.log_prompt(generator=generator, name="Test Generator")
        self.assertTrue("Test Generator" in prompt_logger._trace_map)

        # Update the prompt variable and value
        preset_prompt_kwargs = {"input_str": "Hello, updated world!"}
        generator = Generator(
            model_client=self.mock_api_client, preset_prompt_kwargs=preset_prompt_kwargs
        )
        prompt_logger.log_prompt(generator=generator, name="Test Generator")
        self.assertEqual(
            prompt_logger._trace_map["Test Generator"][1].prompt_states[
                "preset_prompt_kwargs"
            ]["input_str"],
            "Hello, updated world!",
        )

        # update the template
        template = "Hello, {{ input_str }}!"
        generator = Generator(model_client=self.mock_api_client, template=template)
        prompt_logger.log_prompt(generator=generator, name="Test Generator")
        self.assertEqual(
            prompt_logger._trace_map["Test Generator"][2].prompt_states[
                "_template_string"
            ],
            "Hello, {{ input_str }}!",
        )
        self._clean_up()
