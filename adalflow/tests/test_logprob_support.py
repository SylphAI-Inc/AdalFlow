import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from adalflow.core.generator import Generator
from adalflow.core.model_client import ModelClient
from adalflow.core.types import GeneratorOutput, TokenLogProb, ModelType
from adalflow.components.model_client.openai_client import OpenAIClient


class _DummyLogprobClient(ModelClient):
    """Minimal ModelClient stub that reports logprob invocations."""

    def __init__(self):
        super().__init__()
        self.last_input = None
        self.last_model_kwargs = None

    # The following abstract methods are unused in the tests but must be defined.
    def convert_inputs_to_api_kwargs(self, input=None, model_kwargs=None, model_type=None):
        return {"input": input, "model_kwargs": model_kwargs, "model_type": model_type}

    def call(self, api_kwargs=None, model_type=ModelType.UNDEFINED):
        raise AssertionError("call() should not be used in logprob path")

    def parse_chat_completion(self, completion):
        return GeneratorOutput(data=completion)

    def track_completion_usage(self, *args, **kwargs):
        return None

    def list_models(self):
        return []

    def call_with_logprobs(self, input="", model_kwargs=None, model_type=ModelType.UNDEFINED):
        self.last_input = input
        self.last_model_kwargs = model_kwargs or {}

        message = SimpleNamespace(content="positive")
        choice = SimpleNamespace(message=message, index=0)

        logprobs = [
            [
                TokenLogProb(token="positive", logprob=-0.1, choice_index=0),
                TokenLogProb(token="negative", logprob=-5.0, choice_index=0),
            ]
        ]

        completion = SimpleNamespace(choices=[choice])
        return completion, logprobs


class _DummyNoLogprobClient(ModelClient):
    """ModelClient stub that does not implement logprob support."""

    def convert_inputs_to_api_kwargs(self, input=None, model_kwargs=None, model_type=None):
        return {"input": input, "model_kwargs": model_kwargs, "model_type": model_type}

    def call(self, api_kwargs=None, model_type=ModelType.UNDEFINED):
        return "fallback"

    def parse_chat_completion(self, completion):
        return GeneratorOutput(data="fallback")

    def track_completion_usage(self, *args, **kwargs):
        return None

    def list_models(self):
        return []

    def call_with_logprobs(self, *args, **kwargs):
        raise NotImplementedError


class LogprobIntegrationTests(unittest.TestCase):
    def test_select_from_options_uses_logprob_path_when_available(self):
        client = _DummyLogprobClient()
        generator = Generator(
            model_client=client,
            model_kwargs={"model": "fake-model", "temperature": 0.2},
            template="Classify sentiment: {{input_str}}",
        )

        with patch.object(generator, "call", wraps=generator.call) as call_mock:
            result = generator.select_from_options(
                options=["positive", "negative"],
                prompt_kwargs={"input_str": "I love this!"},
                model_kwargs={"max_tokens": 42},
            )

        self.assertEqual("positive", result)
        call_mock.assert_not_called()

        # The logprob client should see the merged prompt and kwargs.
        self.assertIn("positive", client.last_input)
        self.assertEqual(client.last_model_kwargs["model"], "fake-model")
        self.assertEqual(client.last_model_kwargs["temperature"], 0.2)
        self.assertEqual(client.last_model_kwargs["max_tokens"], 42)

        # Template must be restored after the call.
        self.assertEqual(generator.template, "Classify sentiment: {{input_str}}")

    def test_select_from_options_falls_back_when_logprob_missing(self):
        client = _DummyNoLogprobClient()
        generator = Generator(model_client=client, template="Pick: {{input_str}}")

        with patch.object(
            generator,
            "call",
            return_value=GeneratorOutput(data="negative"),
        ) as call_mock:
            result = generator.select_from_options(
                options=["positive", "negative"],
                prompt_kwargs={"input_str": "Feels off"},
            )

        self.assertEqual("negative", result)
        call_mock.assert_called_once()


class OpenAIClientLogprobTests(unittest.TestCase):
    def test_extract_logprobs_attaches_choice_index(self):
        client = OpenAIClient.__new__(OpenAIClient)

        token_entries = [
            SimpleNamespace(token="positive", logprob=-0.1),
            SimpleNamespace(token="!", logprob=-0.05),
        ]
        choice = SimpleNamespace(
            index=3,
            logprobs=SimpleNamespace(content=token_entries),
        )
        completion = SimpleNamespace(choices=[choice])

        extracted = client._extract_logprobs(completion)
        self.assertEqual(1, len(extracted))
        self.assertEqual(2, len(extracted[0]))
        self.assertTrue(all(token.choice_index == 3 for token in extracted[0]))

