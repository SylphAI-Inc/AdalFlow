import os
from typing import Dict, Optional, Any
import backoff
import logging
import warnings

from lightrag.utils.lazy_import import safe_import, OptionalPackages

ollama = safe_import(OptionalPackages.OLLAMA.value[0], OptionalPackages.OLLAMA.value[1])
import ollama
from ollama import RequestError, GenerateResponse


from lightrag.core.model_client import ModelClient
from lightrag.core.types import ModelType, EmbedderOutput, Embedding

log = logging.getLogger(__name__)


class OllamaClient(ModelClient):
    __doc__ = r"""A component wrapper for the Ollama SDK client.

    To make a model work, you need to:
    - [Download Ollama app] Go to https://github.com/ollama/ollama?tab=readme-ov-file to download the Ollama app (command line tool).
      Choose the appropriate version for your operating system.

    - [Pull a model] Run the following command to pull a model:

            ```shell
            ollama pull llama3
            ```
    - [Run a model] Run the following command to run a model:

        ```shell
        ollama run llama3
        ```

        This model will be available at http://localhost:11434. You can also chat with the model at the terminal after running the command.

    -


    Visit https://github.com/ollama/ollama-python for more SDK information details.

    References:
    -

    Tested Ollama models: 7/9/24
    -  internlm2:latest
    -  jina/jina-embeddings-v2-base-en:latest

    .. note::
    """

    def __init__(self, host: Optional[str] = None):
        r"""

        Args:
            host (Optional[str], optional): Optional host URI.
                If not provided, it will look for OLLAMA_HOST env variable. Defaults to None.
                The default host is "http://localhost:11434".
        """
        super().__init__()

        self._host = host or os.getenv("OLLAMA_HOST")
        if not self._host:
            warnings.warn(
                "Better to provide host or set OLLAMA_HOST env variable, we will use the default host"
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

    def parse_embedding_response(self, response: Dict[str, float]) -> EmbedderOutput:
        r"""Parse the embedding response to a structure LightRAG components can understand.
        Pull the embedding from response['embedding'] and store it Embedding dataclass
        """
        try:
            embeddings = Embedding(embedding=response["embedding"], index=None)
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
