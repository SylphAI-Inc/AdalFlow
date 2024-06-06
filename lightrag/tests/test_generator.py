import pytest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch, Mock
import os
import shutil

from lightrag.core.data_classes import GeneratorOutput
from lightrag.core.generator import Generator
from lightrag.components.api_client import OpenAIClient
from lightrag.tracing import GeneratorStateLogger
import lightrag.utils.setup_env


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
        self.assertEqual(output.data, "Generated text response")

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
            model_client=self.mock_api_client, preset_prompt_kwargs=preset_prompt_kwargs
        )

        prompt_logger.log_prompt(generator=generator, name="Test Generator")
        print(
            f"""preset_prompt_kwargs: {prompt_logger._trace_map["Test Generator"][-1].prompt_states}"""
        )
        self.assertEqual(
            prompt_logger._trace_map["Test Generator"][1].prompt_states["data"][
                "preset_prompt_kwargs"
            ]["input_str"],
            "Hello, updated world!",
        )

        # update the template
        template = "Hello, {{ input_str }}!"
        generator = Generator(model_client=self.mock_api_client, template=template)
        prompt_logger.log_prompt(generator=generator, name="Test Generator")
        self.assertEqual(
            prompt_logger._trace_map["Test Generator"][2].prompt_states["data"][
                "_template_string"
            ],
            "Hello, {{ input_str }}!",
        )
        self._clean_up()
