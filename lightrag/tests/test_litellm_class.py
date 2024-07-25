import unittest
from unittest.mock import Mock
from lightrag.core.types import ModelType
from lightrag.components.model_client.lite_client import LiteClient



from lightrag.utils import setup_env # ensure you have .env with OPENAI_API_KEY
setup_env("C:/Users/jean\Documents/molo/LightRAG/.env")  # need to setup env

class test_lite_model(unittest.TestCase):
    
    def test_lite_client(self):
        litellm_client=Mock(spec=LiteClient(model="groq/llama3-70b-8192"))
        print("testing litellm client")
        kwargs = {
           #model already defined in the init , add additionnal argument if needed
           "model": "groq/llama3-70b-8192",
                      
        }
        api_kwargs=litellm_client.convert_inputs_to_api_kwargs(
            input="Hello ?",
            model_kwargs=kwargs,
            model_type=ModelType.LLM,
        ).return_value = {"model":  "groq/llama3-70b-8192", "messages": [{"role": "user", "content": "Hello ?"}]}
        
        assert api_kwargs == {"model":  "groq/llama3-70b-8192", "messages": [{"role": "user", "content": "Hello ?"}]}
        
        output=litellm_client.call(
            api_kwargs=api_kwargs, model_type=ModelType.LLM
        ).return_value = {"message": "Hello"}
        
        assert output == {"message": "Hello"}
        
    def test_lite_client_embeddings(self):
        litellm_client=Mock(spec=LiteClient(model="text-embedding-ada-002-v2"))
        print("testing litellm client")
        kwargs = {
            #model already defined in the init , add additionnal argument if needed
            "model": "text-embedding-ada-002-v2",
                        
        }
        api_kwargs=litellm_client.convert_inputs_to_api_kwargs(
            input="Hello ?",
            model_kwargs=kwargs,
            model_type=ModelType.EMBEDDER,
        ).return_value = {"model":  "text-embedding-ada-002-v2", "input": "Helo"}
        
        assert api_kwargs == {"model":  "text-embedding-ada-002-v2", "input": "Helo"}
        
        output=litellm_client.call(
            api_kwargs=api_kwargs, model_type=ModelType.EMBEDDER
        ).return_value = {"message": "Hello"}
        
       
            
            
        
if __name__ == "__main__":
    unittest.main()
