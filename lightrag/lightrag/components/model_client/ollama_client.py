import os
from typing import Dict, Optional, Any
import backoff
import logging

from lightrag.utils.lazy_import import safe_import, OptionalPackages

ollama = safe_import(OptionalPackages.OLLAMA.value[0], OptionalPackages.OLLAMA.value[1])
from ollama import RequestError, ChatResponse


from lightrag.core.model_client import ModelClient
from lightrag.core.types import ModelType, EmbedderOutput, Embedding

log = logging.getLogger(__name__)


class OllamaClient(ModelClient):
    __doc__ = r"""A component wrapper for the Ollama SDK client.

    Visit https://github.com/ollama/ollama-python for more SDK information details.

    References:
    - 
    
    Tested Ollama models: 7/9/24
    -  internlm2:latest
    -  jina/jina-embeddings-v2-base-en:latest

    .. note::
    """

    def __init__(self, host: Optional[str] = "http://localhost:11434"):
        r"""

        Args:
            host (Optional[str], optional): Optional host URI. Defaults to locally running ollama instance.
        """
        super().__init__()

        self._host = host
        self.init_sync_client()
        self.async_client = None  # only initialize if the async call is called

    def init_sync_client(self):
        """Create the synchronous client"""
        host = self._host or os.getenv("OLLAMA_HOST")
        if not host:
            raise ValueError("Must provide host or set OLLAMA_HOST env variable")
        self.sync_client = ollama.Client(host=host)

    def init_async_client(self):
        """Create the asynchronous client"""
        host = self._host or os.getenv("OLLAMA_HOST")
        if not host:
            raise ValueError("Must provide host or set OLLAMA_HOST env variable")
        self.async_client = ollama.AsyncClient(host=host)

    def parse_chat_completion(self, completion: ChatResponse) -> Any:
        """Parse the completion to a str."""
        log.debug(f"completion: {completion}")
        return completion["message"]["content"]

    def parse_embedding_response(self, response: Dict[str, float]) -> EmbedderOutput:
        r"""Parse the embedding response to a structure LightRAG components can understand.
        Pull the embedding from response['embedding'] and store it Embedding dataclass
        """
        try:
            embeddings = Embedding(embedding=response["embedding"])
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
        r"""
        For LLM, expect model_kwargs to have the following keys:
         model: str,
         prompt: str,

        For EMBEDDER, expect model_kwargs to have the following keys:
         model: str,
         prompt: str,
        """
        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            # make sure input is a string
            if input is not None and input != "":
                final_model_kwargs["prompt"] = input
                return final_model_kwargs
            else:
                raise ValueError("input must be text")
        if model_type == ModelType.LLM:
            if input is not None and input != "":
                final_model_kwargs["prompt"] = input
                return final_model_kwargs
            else:
                raise ValueError("Input must be text")
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (RequestError,),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        r"""
        kwargs is the combined input and model_kwargs
        """
        if "model" not in api_kwargs:
            raise ValueError("model must be specified")
        log.info(f"api_kwargs: {api_kwargs}")
        if model_type == ModelType.EMBEDDER:
            return self.sync_client.embeddings(**api_kwargs)
        if model_type == ModelType.LLM:
            return self.sync_client.generate(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (RequestError),
        max_time=5,
    )
    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        """
        kwargs is the combined input and model_kwargs
        """
        if self.async_client is None:
            self.init_async_client()
        if "model" not in api_kwargs:
            raise ValueError("model must be specified")
        if model_type == ModelType.EMBEDDER:
            return await self.async_client.embeddings(**api_kwargs)
        if model_type == ModelType.LLM:
            return await self.async_client.generate(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")
