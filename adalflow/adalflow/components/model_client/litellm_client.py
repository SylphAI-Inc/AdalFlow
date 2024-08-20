"""LiteLLM ModelClient integration."""

import os
from typing import Dict, Optional, Any, List
import backoff
import logging
import warnings
from typing import (
    Dict,
    Optional,
    List,
    Any,
    Generator,
    Union,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletion
from adalflow.core.types import (
    ModelType,
    EmbedderOutput,
    GeneratorOutput,
)
from adalflow.utils.lazy_import import safe_import, OptionalPackages
from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, EmbedderOutput, Embedding

# Conditional import of litellm
litellm = safe_import(
    OptionalPackages.LITELLM.value[0], OptionalPackages.LITELLM.value[1]
)

from litellm import validate_environment
from litellm.exceptions import (
    Timeout,
    InternalServerError,
    APIConnectionError,
    RateLimitError,
)

log = logging.getLogger(__name__)


# completion parsing functions and you can combine them into one singple chat completion parser
def get_first_message_content(completion: ChatCompletion) -> str:
    r"""When we only need the content of the first message.
    It is the default parser for chat completion."""
    return completion.choices[0].message.content


class LiteLLMClient(ModelClient):
    """A component wrapper for the LiteLLM API client.

    Visit https://www.litellm.ai/ for more api details.
    Check https://docs.litellm.ai/docs/providers for the available models.

    Tested LiteLLM models: 07/19/2024
    - openai/gpt-4o
    - anthropic/claude3-haiku-20240307
    - ollama/llama3.1:latest
    - openai/text-embedding-ada-002
    """

    def __init__(self, api_key: Optional[str] = None):
        r"""It is recommended to set the API_KEY for your model in the environment variable instead of passing it as an argument.
        See: https://docs.litellm.ai/docs/
            - Per getting started

        Args:
            api_key (Optional[str], optional): Model provider API Key. Defaults to None.
        """
        super().__init__()
        self._api_key = api_key
        self.sync_client = self.init_sync_client()
        self.async_client = None  # only initialize if the async call is called

    def _validate_environment(self, model: str):
        isValid = validate_environment(model)
        if not isValid["keys_in_environment"]:
            env_api_key = next(iter(isValid["missing_keys"]), "API_KEY")
            if self._api_key is None:
                raise ValueError(f"Missing {env_api_key} in environment.")
            else:
                warnings.warn(
                    f"Best to set the {env_api_key} environment variable, we'll set it for now."
                )
                os.environ[env_api_key] = self._api_key

    def init_sync_client(self) -> Any:
        return litellm

    def init_async_client(self) -> Any:
        return litellm

    def parse_chat_completion(
        self,
        completion: Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]],
    ) -> "GeneratorOutput":
        """Parse the completion, and put it into the raw_response."""
        log.debug(f"completion: {completion}")
        try:
            data = completion.choices[0].message.content
            # usage = self.track_completion_usage(completion)
            return GeneratorOutput(data=None, error=None, raw_response=data, usage=None)
        except Exception as e:
            log.error(f"Error parsing the completion: {e}")
            return GeneratorOutput(data=None, error=str(e), raw_response=completion)

    def parse_embedding_response(
        self, response: Dict[str, List[float]]
    ) -> EmbedderOutput:
        """Parse the embedding response to a structure LightRAG components can understand."""
        try:
            embeddings = Embedding(embedding=response["data"][0]["embedding"], index=0)
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
        if "model" not in model_kwargs:
            raise ValueError("model must be specified")
        self._validate_environment(model_kwargs["model"])
        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            if not isinstance(input, List):
                raise TypeError("input must be a string or a list of strings")
            final_model_kwargs["input"] = input
        elif model_type == ModelType.LLM:
            messages: List[Dict[str, str]] = []
            if input is not None and input != "":
                # By this point we should have the model in the args, handle special case for Anthropic, otherwise go with adding to the system role.
                # TODO: Better handling for models that don't accept system role as first message.
                if "claude" in model_kwargs["model"].lower():
                    messages.append(
                        {
                            "role": "user",
                            "content": input,
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "system",
                            "content": input,
                        }
                    )
            final_model_kwargs["messages"] = messages
        else:
            raise ValueError(f"model_type {model_type} is not supported")

        return final_model_kwargs

    @backoff.on_exception(
        backoff.expo,
        (Timeout, APIConnectionError, InternalServerError, RateLimitError),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):

        self._validate_environment(api_kwargs["model"])
        log.info(f"api_kwargs: {api_kwargs}")
        if not self.sync_client:
            self.init_sync_client()
        if self.sync_client is None:
            raise RuntimeError("Sync client is not initialized")
        if model_type == ModelType.EMBEDDER:
            return self.sync_client.embedding(**api_kwargs)
        if model_type == ModelType.LLM:
            return self.sync_client.completion(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (Timeout, APIConnectionError, InternalServerError, RateLimitError),
        max_time=5,
    )
    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        """Asynchronous call to the API"""
        # if "model" not in api_kwargs or api_kwargs['model'] == "":
        #     raise ValueError("model must be specified")
        # self._validate_environment(api_kwargs["model"])
        self._validate_environment(api_kwargs["model"])
        log.info(f"api_kwargs: {api_kwargs}")
        if self.async_client is None:
            self.async_client = self.init_async_client()
        if model_type == ModelType.EMBEDDER:
            return await self.async_client.aembedding(**api_kwargs)
        elif model_type == ModelType.LLM:
            return await self.async_client.acompletion(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")
