import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock

from adalflow.core.types import ModelType, EmbedderOutput, GeneratorOutput
from adalflow.components.model_client.twelvelabs_client import (
    TwelveLabsClient,
    DEFAULT_EMBEDDER_MODEL,
    DEFAULT_LLM_MODEL,
)


def _fake_embedding_response(vec, model_name=DEFAULT_EMBEDDER_MODEL):
    segment = SimpleNamespace(float_=vec)
    text_embedding = SimpleNamespace(segments=[segment])
    return SimpleNamespace(model_name=model_name, text_embedding=text_embedding)


def _fake_analyze_response():
    usage = SimpleNamespace(input_tokens=20, output_tokens=10)
    return SimpleNamespace(
        id="task-123",
        data="A dog plays fetch in a sunny park.",
        finish_reason="stop",
        usage=usage,
        error=None,
    )


class TestTwelveLabsClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = TwelveLabsClient(api_key="fake_api_key")

    # ----------------------------------------------------------- input mapping
    def test_convert_inputs_embedder(self):
        api_kwargs = self.client.convert_inputs_to_api_kwargs(
            input="hello world",
            model_kwargs={"model": "marengo3.0"},
            model_type=ModelType.EMBEDDER,
        )
        self.assertEqual(
            api_kwargs, {"model_name": "marengo3.0", "texts": ["hello world"]}
        )

    def test_convert_inputs_embedder_default_model_and_list(self):
        api_kwargs = self.client.convert_inputs_to_api_kwargs(
            input=["a", "b"],
            model_kwargs={},
            model_type=ModelType.EMBEDDER,
        )
        self.assertEqual(api_kwargs["model_name"], DEFAULT_EMBEDDER_MODEL)
        self.assertEqual(api_kwargs["texts"], ["a", "b"])

    def test_convert_inputs_llm_video_url(self):
        api_kwargs = self.client.convert_inputs_to_api_kwargs(
            input="What happens in this video?",
            model_kwargs={"video_url": "https://example.com/v.mp4", "max_tokens": 512},
            model_type=ModelType.LLM,
        )
        self.assertEqual(api_kwargs["model_name"], DEFAULT_LLM_MODEL)
        self.assertEqual(api_kwargs["prompt"], "What happens in this video?")
        self.assertEqual(
            api_kwargs["video"], {"type": "url", "url": "https://example.com/v.mp4"}
        )
        self.assertEqual(api_kwargs["max_tokens"], 512)

    def test_convert_inputs_llm_video_id(self):
        api_kwargs = self.client.convert_inputs_to_api_kwargs(
            input="Summarize",
            model_kwargs={"video_id": "vid-1"},
            model_type=ModelType.LLM,
        )
        self.assertEqual(api_kwargs["video_id"], "vid-1")
        self.assertNotIn("video", api_kwargs)

    def test_convert_inputs_llm_requires_video(self):
        with self.assertRaises(ValueError):
            self.client.convert_inputs_to_api_kwargs(
                input="hi", model_kwargs={}, model_type=ModelType.LLM
            )

    def test_convert_inputs_unsupported_type(self):
        with self.assertRaises(ValueError):
            self.client.convert_inputs_to_api_kwargs(
                input="hi", model_kwargs={}, model_type=ModelType.RERANKER
            )

    # ---------------------------------------------------------------- parsing
    def test_parse_embedding_response(self):
        responses = [
            _fake_embedding_response([0.1, 0.2, 0.3]),
            _fake_embedding_response([0.4, 0.5, 0.6]),
        ]
        output = self.client.parse_embedding_response(responses)
        self.assertIsInstance(output, EmbedderOutput)
        self.assertEqual(len(output.data), 2)
        self.assertEqual(output.data[0].embedding, [0.1, 0.2, 0.3])
        self.assertEqual(output.data[0].index, 0)
        self.assertEqual(output.data[1].index, 1)
        self.assertEqual(output.model, DEFAULT_EMBEDDER_MODEL)
        self.assertIsNone(output.error)

    def test_parse_chat_completion(self):
        output = self.client.parse_chat_completion(_fake_analyze_response())
        self.assertIsInstance(output, GeneratorOutput)
        self.assertEqual(output.raw_response, "A dog plays fetch in a sunny park.")
        self.assertEqual(output.id, "task-123")
        self.assertEqual(output.usage.prompt_tokens, 20)
        self.assertEqual(output.usage.completion_tokens, 10)
        self.assertEqual(output.usage.total_tokens, 30)

    # ----------------------------------------------------------------- calling
    def test_call_embedder(self):
        mock_sync = Mock()
        mock_sync.embed.create = Mock(
            side_effect=lambda model_name, text: _fake_embedding_response(
                [0.1, 0.2], model_name
            )
        )
        self.client.sync_client = mock_sync

        api_kwargs = {"model_name": "marengo3.0", "texts": ["a", "b"]}
        result = self.client.call(api_kwargs=api_kwargs, model_type=ModelType.EMBEDDER)
        self.assertEqual(len(result), 2)
        self.assertEqual(mock_sync.embed.create.call_count, 2)

    def test_call_llm(self):
        mock_sync = Mock()
        mock_sync.analyze = Mock(return_value=_fake_analyze_response())
        self.client.sync_client = mock_sync

        api_kwargs = {
            "model_name": "pegasus1.5",
            "prompt": "hi",
            "video_id": "vid-1",
        }
        result = self.client.call(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        mock_sync.analyze.assert_called_once_with(**api_kwargs)
        self.assertEqual(result.data, "A dog plays fetch in a sunny park.")

    async def test_acall_embedder(self):
        self.client.async_client = Mock()
        self.client.async_client.embed.create = AsyncMock(
            return_value=_fake_embedding_response([0.1, 0.2])
        )
        api_kwargs = {"model_name": "marengo3.0", "texts": ["a"]}
        result = await self.client.acall(
            api_kwargs=api_kwargs, model_type=ModelType.EMBEDDER
        )
        self.assertEqual(len(result), 1)


@unittest.skipUnless(
    os.getenv("TWELVELABS_API_KEY"),
    "TWELVELABS_API_KEY not set; skipping live TwelveLabs integration test",
)
class TestTwelveLabsClientLive(unittest.TestCase):
    def test_marengo_embedding_dim(self):
        from adalflow.core import Embedder

        embedder = Embedder(
            model_client=TwelveLabsClient(),
            model_kwargs={"model": DEFAULT_EMBEDDER_MODEL},
        )
        output = embedder("A dog playing in the park")
        self.assertIsNone(output.error)
        self.assertEqual(len(output.data[0].embedding), 512)


if __name__ == "__main__":
    unittest.main()
