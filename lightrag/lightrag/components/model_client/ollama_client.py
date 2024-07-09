"""Cohere ModelClient integration."""

import os
from typing import Dict, Sequence, Optional, List, Any, TypeVar, Callable
import backoff
import logging

from lightrag.utils.lazy_import import safe_import, OptionalPackages

cohere = safe_import(OptionalPackages.OLLAMA.value[0], OptionalPackages.OLLAMA.value[1])
from ollama import (
    RequestError, ChatResponse
)


from lightrag.core.model_client import ModelClient
from lightrag.core.types import ModelType, EmbedderOutput, Embedding
import ollama


log = logging.getLogger(__name__)

class OllamaClient(ModelClient):
    __doc__ = r"""A component wrapper for the Ollama SDK client.

    Visit https://github.com/ollama/ollama-python for more SDK information details.

    References:
    - 
    
    Tested Ollama models: 7/9/24
    -  

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
        host = self._host or os.getenv("OLLAMA_HOST")
        if not host:
            raise ValueError("Must provide host or set OLLAMA_HOST env variable")
        self.sync_client = ollama.Client(host=host)

    def init_async_client(self):
        host = self._host or os.getenv("OLLAMA_HOST")
        if not host:
            raise ValueError("Must provide host or set OLLAMA_HOST env variable")
        self.async_client = ollama.AsyncClient(host=host)

    def parse_chat_completion(self, completion: ChatResponse) -> Any:
        """Parse the completion to a str."""
        log.debug(f"completion: {completion}")
        return completion['message']['content']

    def parse_embedding_response(
        self, response: Dict[str, float]
    ) -> EmbedderOutput:
        r"""Parse the embedding response to a structure LightRAG components can understand.

        Should be called in ``Embedder``.
        """
        try:
            embeddings = Embedding(embedding=response['embedding'])
            return EmbedderOutput(data=[embeddings])
        except Exception as e:
            log.error(f"Error parsing the embedding response: {e}")
            return EmbedderOutput(data=[], error=str(e), raw_response=response)

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,  # for retriever, it is a list of string.
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        r"""
        For LLM, expect model_kwargs to have the following keys:
         model: str,
         messages: str,
        """
        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            # make sure input is a string
            assert isinstance(input, str), "input must be a sequence of text"
            final_model_kwargs["prompt"] = input
        if model_type == ModelType.LLM:
            messages: List[Dict[str, str]] = []
            if input is not None and input != "":
                messages.append({"role": "user", "content": input})
            assert isinstance(
                messages, Sequence
            ), "input must be a sequence of messages"
            final_model_kwargs["messages"] = messages
        else:
            raise ValueError(f"model_type {model_type} is not supported")
        return final_model_kwargs

    @backoff.on_exception(
        backoff.expo,
        (RequestError,),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        assert (
            "model" in api_kwargs
        ), f"model must be specified in api_kwargs: {api_kwargs}"
        log.info(f"api_kwargs: {api_kwargs}")
        if model_type == ModelType.EMBEDDER:
            return self.sync_client.embeddings(**api_kwargs)
        if model_type == ModelType.LLM:
            return self.sync_client.chat(**api_kwargs)
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
        if self.async_client is None:
            self.init_async_client()
        assert "model" in api_kwargs, "model must be specified"
        if model_type == ModelType.EMBEDDER:
            return await self.async_client.embeddings(**api_kwargs)
        if model_type == ModelType.LLM:
            return await self.async_client.chat(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")
