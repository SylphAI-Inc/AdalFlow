import os
from typing import Dict, Optional, Any, TypeVar, List, Type
import backoff
import logging
import warnings

from lightrag.utils.lazy_import import safe_import, OptionalPackages

ollama = safe_import(OptionalPackages.OLLAMA.value[0], OptionalPackages.OLLAMA.value[1])
import ollama
from ollama import RequestError, ResponseError, GenerateResponse


from lightrag.core.model_client import ModelClient
from lightrag.core.types import ModelType, EmbedderOutput, Embedding

log = logging.getLogger(__name__)

T = TypeVar("T")


class OllamaClient(ModelClient):
    __doc__ = r"""A component wrapper for the Ollama SDK client.

    To make a model work, you need to:

    - [Download Ollama app] Go to https://github.com/ollama/ollama?tab=readme-ov-file to download the Ollama app (command line tool).
      Choose the appropriate version for your operating system.

    - [Pull a model] Run the following command to pull a model:

    .. code-block:: shell

        ollama pull llama3

    - [Run a model] Run the following command to run a model:

    .. code-block:: shell

        ollama run llama3

    This model will be available at http://localhost:11434. You can also chat with the model at the terminal after running the command.

    Args:
        host (Optional[str], optional): Optional host URI.
            If not provided, it will look for OLLAMA_HOST env variable. Defaults to None.
            The default host is "http://localhost:11434".

    References:

        - https://github.com/ollama/ollama-python
        - https://github.com/ollama/ollama
        - Models: https://ollama.com/library

    Tested Ollama models: 7/9/24

    -  internlm2:latest
    -  llama3
    -  jina/jina-embeddings-v2-base-en:latest

    .. note::
       We use `embeddings` and `generate` apis from Ollama SDK.
       Please refer to https://github.com/ollama/ollama-python/blob/main/ollama/_client.py for model_kwargs details.
    """

    def __init__(self, host: Optional[str] = None):
        super().__init__()

        self._host = host or os.getenv("OLLAMA_HOST")
        if not self._host:
            warnings.warn(
                "Better to provide host or set OLLAMA_HOST env variable. We will use the default host http://localhost:11434 for now."
            )
            self._host = "http://localhost:11434"

        log.debug(f"Using host: {self._host}")

        self.init_sync_client()
        self.async_client = None  # only initialize if the async call is called

    def init_sync_client(self):
        """Create the synchronous client"""

        self.sync_client = ollama.Client(host=self._host)

    def init_async_client(self):
        """Create the asynchronous client"""

        self.async_client = ollama.AsyncClient(host=self._host)

    def parse_chat_completion(self, completion: GenerateResponse) -> Any:
        """Parse the completion to a str. We use the generate with prompt instead of chat with messages."""
        log.debug(f"completion: {completion}")
        if "response" in completion:
            return completion["response"]
        else:
            log.error(f"Error parsing the completion: {completion}")
            raise ValueError(f"Error parsing the completion: {completion}")

    def parse_embedding_response(
        self, response: Dict[str, List[float]]
    ) -> EmbedderOutput:
        r"""Parse the embedding response to a structure LightRAG components can understand.
        Pull the embedding from response['embedding'] and store it Embedding dataclass
        """
        try:
            embeddings = Embedding(embedding=response["embedding"], index=0)
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
        # TODO: ollama will support batch embedding in the future: https://ollama.com/blog/embedding-models
        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                final_model_kwargs["prompt"] = input
                return final_model_kwargs
            else:
                raise ValueError(
                    "Ollama does not support batch embedding yet. It only accepts a single string for input for now. Make sure you are not passing a list of strings"
                )
        elif model_type == ModelType.LLM:
            if input is not None and input != "":
                final_model_kwargs["prompt"] = input
                return final_model_kwargs
            else:
                raise ValueError("Input must be text")
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (RequestError, ResponseError),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        if "model" not in api_kwargs:
            raise ValueError("model must be specified")
        log.info(f"api_kwargs: {api_kwargs}")
        if not self.sync_client:
            self.init_sync_client()
        if self.sync_client is None:
            raise RuntimeError("Sync client is not initialized")
        if model_type == ModelType.EMBEDDER:
            return self.sync_client.embeddings(**api_kwargs)
        if model_type == ModelType.LLM:
            return self.sync_client.generate(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (RequestError, ResponseError),
        max_time=5,
    )
    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        if self.async_client is None:
            self.init_async_client()
        if self.async_client is None:
            raise RuntimeError("Async client is not initialized")
        if "model" not in api_kwargs:
            raise ValueError("model must be specified")
        if model_type == ModelType.EMBEDDER:
            return await self.async_client.embeddings(**api_kwargs)
        if model_type == ModelType.LLM:
            return await self.async_client.generate(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @classmethod
    def from_dict(cls: Type["OllamaClient"], data: Dict[str, Any]) -> "OllamaClient":
        obj = super().from_dict(data)
        # recreate the existing clients
        obj.sync_client = obj.init_sync_client()
        obj.async_client = obj.init_async_client()
        return obj

    def to_dict(self, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
        r"""Convert the component to a dictionary."""

        # combine the exclude list
        exclude = list(set(exclude or []) | {"sync_client", "async_client"})

        output = super().to_dict(exclude=exclude)
        return output
