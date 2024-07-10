import unittest
from lightrag.core.types import ModelType
from lightrag.components.model_client import OllamaClient
import ollama

# Check if ollama model is installed
def model_installed(model_name: str):
    model_list = ollama.list()
    for model in model_list['models']:
        if model['name'] == model_name:
            return True
    return False


class TestOllamaModelClient(unittest.TestCase):
    
    def test_ollama_llm_client(self):
        ollama_client = OllamaClient()
        print("Testing ollama LLM client")
        # run the model
        kwargs = {
            "model": "qwen2:0.5b",
        }
        api_kwargs = ollama_client.convert_inputs_to_api_kwargs(
            input="Hello world",
            model_kwargs=kwargs,
            model_type=ModelType.LLM,
        )
        # check if the model is installed
        if model_installed(kwargs["model"]) is not True:
            ollama.pull(kwargs["model"])

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
        # Check if model is installed
        if model_installed(kwargs["model"]) is not True:
            ollama.pull(kwargs["model"])

        output = ollama_client.call(
            api_kwargs=api_kwargs, model_type=ModelType.EMBEDDER
        )

        print(ollama_client)
        print(f"output: {output}")
    

if __name__ == "__main__":
    unittest.main()
