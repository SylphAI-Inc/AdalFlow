import os
from typing import Dict, Optional, Any, List, Union
import backoff
import logging
import warnings

from lightrag.utils.lazy_import import safe_import, OptionalPackages
from lightrag.core.model_client import ModelClient
from lightrag.core.types import ModelType, EmbedderOutput, Embedding

# Conditional import of litellm
litellm = safe_import(OptionalPackages.LITE.value[0], OptionalPackages.LITE.value[1])

from litellm import completion, embedding, get_supported_openai_params,acompletion,aembedding
from litellm.exceptions import (
    Timeout,
    InternalServerError,
    APIConnectionError,
    RateLimitError,
)

log = logging.getLogger(__name__)

class LiteClient(ModelClient):
    """A component wrapper for the LiteLLM API client.

    Visit https://litelms.com/docs/ for more api details.
    Check https://litelms.com/docs/models for the available models.

    Tested LiteLLM models: 07/19/2024
    - mixtral-8x7b
    - gemma-7b-it
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout: Optional[int] = 600,
        drop_params: bool = False
    ):
        super().__init__()
        self.model = model
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.base_url = base_url
        self.api_version = api_version
        self.timeout = timeout
        self.drop_params = drop_params
        self.sync_client = self.init_sync_client()
        self.async_client = None  # only initialize if the async call is called

    def init_sync_client(self) -> Any:
        import litellm
        validate = litellm.validate_environment(model=self.model)
        if validate["keys_in_environment"]==False:
            ApiError = validate["missing_keys"]
            raise ValueError(f"Environment variable {ApiError} required for LiteLM model {self.model} must be set")
        return litellm 

    def init_async_client(self) -> Any:
        import litellm
        validate = litellm.validate_environment(self.model)
        if validate["keys_in_environment"]==False:
            ApiError = validate["missing_keys"]
            raise ValueError(f"Environment variable {ApiError} must be set")
        return  litellm
    
    def parse_chat_completion(self, response: Any) -> str:
        """Parse completion output to strings"""
        log.debug(f"completion: {response}")
        if "choices" in response:
            return response['choices'][0]['message']['content']
        else:
            log.error(f"Error parsing the completion: {response}")
            raise ValueError(f"Error parsing the completion: {response}")
    
    def parse_embedding_response(self, response: Dict[str, List[float]]) -> EmbedderOutput:
        """Parse the embedding response to a structure LightRAG components can understand."""
        try:
            embeddings = Embedding(embedding=response["data"][0]['embedding'], index=0)
            return EmbedderOutput(data=[embeddings])
        except Exception as e:
            log.error(f"Error parsing the embedding response: {e}")
            return EmbedderOutput(data=[], error=str(e), raw_response=response)
        
    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """Convert the Component's standard input and model_kwargs into API-specific format"""
        final_model_kwargs = model_kwargs.copy()
        supported_params = get_supported_openai_params(self.model)

        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            if not isinstance(input, List):
                raise TypeError("input must be a string or a list of strings")
            final_model_kwargs["input"] = input
        elif model_type == ModelType.LLM:
            messages: List[Dict[str, str]] = []
            if input is not None and input != "":
                messages.append({"role": "user","content": input,})
            final_model_kwargs["messages"] = messages
        else:
            raise ValueError(f"model_type {model_type} is not supported")

        # Filter out unsupported params
        if self.drop_params:
            final_model_kwargs = {k: v for k, v in final_model_kwargs.items() if k in supported_params}

        return final_model_kwargs
    
    @backoff.on_exception(
        backoff.expo,
        (Timeout, APIConnectionError, InternalServerError, RateLimitError),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        if "model" not in api_kwargs:
            api_kwargs["model"] = self.model
        log.info(f"api_kwargs: {api_kwargs}")
        if not self.sync_client:
            self.init_sync_client()
        if self.sync_client is None:
            raise RuntimeError("Sync client is not initialized")
        if model_type == ModelType.EMBEDDER:
            return litellm.embedding(**api_kwargs)
        if model_type == ModelType.LLM:
            return self.sync_client.completion(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")
     
    @backoff.on_exception(
        backoff.expo,
        (Timeout, APIConnectionError, InternalServerError, RateLimitError),
        max_time=5,
    )   
    async def acall(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """Asynchronous call to the API"""
        import litellm
        if self.async_client is None:
            self.async_client =  self.init_async_client()
        if model_type == ModelType.EMBEDDER:
            return await litellm.aembedding(**api_kwargs)
        elif model_type == ModelType.LLM:
            return await self.async_client.acompletion(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    def get_supported_openai_params(self, model: Optional[str] = None) -> List[str]:
        """Get supported OpenAI parameters for the specified model"""
        return get_supported_openai_params(model or self.model)