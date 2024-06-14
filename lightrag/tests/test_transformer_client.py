import unittest
from lightrag.components.model_client import (
    TransformersClient,
    TransformerReranker,
    TransformerEmbedder,
    ModelType,
)


class TestTransformerModelClient(unittest.TestCase):
    def setUp(self) -> None:
        self.reranker_input = [
            ["what is panda?", "hi"],
            [
                "what is panda?",
                "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
            ],
        ]

    def test_transformer_embedder(self):
        transformer_embedder_model = "thenlper/gte-base"
        transformer_embedder_model_component = TransformerEmbedder()
        print(
            f"Testing transformer embedder with model {transformer_embedder_model_component}"
        )
        print("Testing transformer embedder")
        output = transformer_embedder_model_component(
            model=transformer_embedder_model, input="Hello world"
        )
        print(output)

    def test_transformer_client(self):
        transformer_client = TransformersClient()
        print("Testing transformer client")
        # run the model
        kwargs = {
            "model": "thenlper/gte-base",
            "mock": False,
        }
        api_kwargs = transformer_client.convert_inputs_to_api_kwargs(
            input="Hello world",
            model_kwargs=kwargs,
            model_type=ModelType.EMBEDDER,
        )
        print(api_kwargs)
        output = transformer_client.call(
            api_kwargs=api_kwargs, model_type=ModelType.EMBEDDER
        )

        print(transformer_client)
        print(output)

    def test_transformer_reranker(self):
        transformer_reranker_model = "BAAI/bge-reranker-base"
        transformer_reranker_model_component = TransformerReranker()
        print(
            f"Testing transformer reranker with model {transformer_reranker_model_component}"
        )

        model_kwargs = {
            "model": transformer_reranker_model,
            "input": self.reranker_input,
        }

        output = transformer_reranker_model_component(
            **model_kwargs,
        )
        # assert output is a list of float with length 2
        self.assertEqual(len(output), 2)
        self.assertEqual(type(output[0]), float)

    def test_transformer_reranker_client(self):
        transformer_reranker_client = TransformersClient(
            model_name="BAAI/bge-reranker-base"
        )
        print("Testing transformer reranker client")
        # run the model
        kwargs = {
            "model": "BAAI/bge-reranker-base",
            "mock": False,
        }
        api_kwargs = transformer_reranker_client.convert_inputs_to_api_kwargs(
            input=self.reranker_input,
            model_kwargs=kwargs,
            model_type=ModelType.RERANKER,
        )
        print(api_kwargs)
        self.assertEqual(api_kwargs["model"], "BAAI/bge-reranker-base")
        output = transformer_reranker_client.call(
            api_kwargs=api_kwargs, model_type=ModelType.RERANKER
        )
        self.assertEqual(type(output), list)
