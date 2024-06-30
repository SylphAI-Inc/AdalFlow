import unittest
import torch

from lightrag.components.model_client import (
    TransformersClient,
    TransformerReranker,
    TransformerLLM,
    TransformerEmbedder,
)
from lightrag.core.types import ModelType

# Set the number of threads for PyTorch, avoid segementation fault
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class TestTransformerModelClient(unittest.TestCase):
    def setUp(self) -> None:

        self.query = "what is panda?"
        self.documents = [
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
            "The red panda (Ailurus fulgens), also called the lesser panda, the red bear-cat, and the red cat-bear, is a mammal native to the eastern Himalayas and southwestern China.",
        ]

    # def test_transformer_embedder(self):
    #     transformer_embedder_model = "thenlper/gte-base"
    #     transformer_embedder_model_component = TransformerEmbedder(
    #         model_name=transformer_embedder_model
    #     )
    #     print(
    #         f"Testing transformer embedder with model {transformer_embedder_model_component}"
    #     )
    #     print("Testing transformer embedder")
    #     output = transformer_embedder_model_component(
    #         model=transformer_embedder_model, input="Hello world"
    #     )
    #     print(output)

    # def test_transformer_client(self):
    #     transformer_client = TransformersClient()
    #     print("Testing transformer client")
    #     # run the model
    #     kwargs = {
    #         "model": "thenlper/gte-base",
    #         # "mock": False,
    #     }
    #     api_kwargs = transformer_client.convert_inputs_to_api_kwargs(
    #         input="Hello world",
    #         model_kwargs=kwargs,
    #         model_type=ModelType.EMBEDDER,
    #     )
    #     # print(api_kwargs)
    #     output = transformer_client.call(
    #         api_kwargs=api_kwargs, model_type=ModelType.EMBEDDER
    #     )

    #     # print(transformer_client)
    #     # print(output)

    # def test_transformer_reranker(self):
    #     transformer_reranker_model = "BAAI/bge-reranker-base"
    #     transformer_reranker_model_component = TransformerReranker()
    #     # print(
    #     #     f"Testing transformer reranker with model {transformer_reranker_model_component}"
    #     # )

    #     model_kwargs = {
    #         "model": transformer_reranker_model,
    #         "documents": self.documents,
    #         "query": self.query,
    #         "top_k": 2,
    #     }

    #     output = transformer_reranker_model_component(
    #         **model_kwargs,
    #     )
    #     # assert output is a list of float with length 2
    #     self.assertEqual(len(output), 2)
    #     self.assertEqual(type(output[0]), float)

    # def test_transformer_reranker_client(self):
    #     transformer_reranker_client = TransformersClient(
    #         model_name="BAAI/bge-reranker-base"
    #     )
    #     print("Testing transformer reranker client")
    #     # run the model
    #     kwargs = {
    #         "model": "BAAI/bge-reranker-base",
    #         "documents": self.documents,
    #         "top_k": 2,
    #     }
    #     api_kwargs = transformer_reranker_client.convert_inputs_to_api_kwargs(
    #         input=self.query,
    #         model_kwargs=kwargs,
    #         model_type=ModelType.RERANKER,
    #     )
    #     print(api_kwargs)
    #     self.assertEqual(api_kwargs["model"], "BAAI/bge-reranker-base")
    #     output = transformer_reranker_client.call(
    #         api_kwargs=api_kwargs, model_type=ModelType.RERANKER
    #     )
    #     self.assertEqual(type(output), tuple)


    # def test_transformer_llm_response(self):
    #     """Test the TransformerLLM model with zephyr-7b-beta for generating a response."""
    #     transformer_llm_model = "HuggingFaceH4/zephyr-7b-beta"
    #     transformer_llm_model_component = TransformerLLM(model_name=transformer_llm_model)
        
    #     # Define a sample input
    #     input_text = "Hello, what's the weather today?"
        
    #     # Test generating a response, providing the 'model' keyword
    #     # response = transformer_llm_model_component(input=input_text, model=transformer_llm_model)
    #     response = transformer_llm_model_component(input_text=input_text)

        
    #     # Check if the response is valid
    #     self.assertIsInstance(response, str, "The response should be a string.")
    #     self.assertTrue(len(response) > 0, "The response should not be empty.")
        
    #     # Optionally, print the response for visual verification during testing
    #     print(f"Generated response: {response}")

        
if __name__ == '__main__':
    unittest.main()