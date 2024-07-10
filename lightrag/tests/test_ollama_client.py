import unittest
from lightrag.core.types import ModelType
from lightrag.components.model_client import OllamaClient


# Set the number of threads for PyTorch, avoid segementation fault


class TestOllamaModelClient(unittest.TestCase):
    
    def test_ollama_llm_client(self):
        ollama_client = OllamaClient()
        print("Testing ollama LLM client")
        # run the model
        kwargs = {
            "model": "internlm2:latest",
        }
        api_kwargs = ollama_client.convert_inputs_to_api_kwargs(
            input="Hello world",
            model_kwargs=kwargs,
            model_type=ModelType.LLM,
        )

        output = ollama_client.call(
            api_kwargs=api_kwargs, model_type=ModelType.LLM
        )

        print(ollama_client)
        print(output)

    def test_ollama_embedding_client(self):
        # jina/jina-embeddings-v2-base-en:latest
        ollama_client = OllamaClient()
        print("Testing ollama embedding client")
        # run the model
        kwargs = {
            "model": "jina/jina-embeddings-v2-base-en:latest",
        }
        api_kwargs = ollama_client.convert_inputs_to_api_kwargs(
            input="Welcome",
            model_kwargs=kwargs,
            model_type=ModelType.EMBEDDER,
        )

        output = ollama_client.call(
            api_kwargs=api_kwargs, model_type=ModelType.EMBEDDER
        )

        print(ollama_client)
        print(f"output: {output}")
    

if __name__ == "__main__":
    unittest.main()
