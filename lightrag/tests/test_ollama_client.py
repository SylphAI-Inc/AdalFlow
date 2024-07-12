import unittest
from unittest.mock import Mock
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
        ollama_client = Mock(spec=OllamaClient())
        print("Testing ollama LLM client")
        # run the model
        kwargs = {
            "model": "qwen2:0.5b",
        }
        api_kwargs = ollama_client.convert_inputs_to_api_kwargs(
            input="Hello world",
            model_kwargs=kwargs,
            model_type=ModelType.LLM,
        ).return_value = {"prompt": "Hello World", "model": "qwen2:0.5b"}
        assert api_kwargs == {"prompt": "Hello World", "model": "qwen2:0.5b"}
        output = ollama_client.call(
            api_kwargs=api_kwargs, model_type=ModelType.LLM
        ).return_value = {'message': "Hello"}
        assert output == {'message': "Hello"}


    def test_ollama_embedding_client(self):
        ollama_client = Mock(spec=OllamaClient())
        print("Testing ollama embedding client")

        # run the model
        kwargs = {
            "model": "jina/jina-embeddings-v2-base-en:latest",
        }
        api_kwargs = ollama_client.convert_inputs_to_api_kwargs(
            input="Welcome",
            model_kwargs=kwargs,
            model_type=ModelType.EMBEDDER,
        ).return_value = {"prompt": "Welcome", "model": "jina/jina-embeddings-v2-base-en:latest"}
        assert api_kwargs == {"prompt": "Welcome", "model": "jina/jina-embeddings-v2-base-en:latest"}
        
        output = ollama_client.call(
            api_kwargs=api_kwargs, model_type=ModelType.EMBEDDER
        ).return_value = {'embedding':[-0.7391586899757385]}
        assert output == {'embedding':[-0.7391586899757385]}
    

if __name__ == "__main__":
    unittest.main()
