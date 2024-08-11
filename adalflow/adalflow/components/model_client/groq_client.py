"""Groq ModelClient integration."""

import os
from typing import Dict, Sequence, Optional, Any, TypeVar
import backoff
import logging
from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, CompletionUsage, GeneratorOutput


from adalflow.utils.lazy_import import safe_import, OptionalPackages

# optional import
groq = safe_import(OptionalPackages.GROQ.value[0], OptionalPackages.GROQ.value[1])


from groq import Groq, AsyncGroq
from groq import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
)
from groq.types.chat import ChatCompletion as GroqChatCompletion
from groq.types import CompletionUsage as GroqCompletionUsage

T = TypeVar("T")
log = logging.getLogger(__name__)


class GroqAPIClient(ModelClient):
    __doc__ = r"""A component wrapper for the Groq API client.

    Visit https://console.groq.com/docs/ for more api details.
    Check https://console.groq.com/docs/models for the available models.

    Tested Groq models: 4/22/2024
    - llama3-8b-8192
    - llama3-70b-8192
    - mixtral-8x7b-32768
    - gemma-7b-it
    """

    def __init__(self, api_key: Optional[str] = None):
        r"""It is recommended to set the GROQ_API_KEY environment variable instead of passing it as an argument.

        Args:
            api_key (Optional[str], optional): Groq API key. Defaults to None.
        """
        super().__init__()
        self._api_key = api_key

        self.sync_client = self.init_sync_client()
        self.async_client = None  # only initialize if the async call is called

    def init_sync_client(self):
        api_key = self._api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Environment variable GROQ_API_KEY must be set")
        return Groq(api_key=api_key)

    def init_async_client(self):
        api_key = self._api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Environment variable GROQ_API_KEY must be set")
        return AsyncGroq(api_key=api_key)

    def parse_chat_completion(
        self, completion: "GroqChatCompletion"
    ) -> "GeneratorOutput":
        """
        Parse the completion to a string output.
        """
        log.debug(f"completion: {completion}")
        try:
            data = completion.choices[0].message.content
            usage = self.track_completion_usage(completion)
            return GeneratorOutput(data=None, usage=usage, raw_response=data)
        except Exception as e:
            log.error(f"Error parsing completion: {e}")
            return GeneratorOutput(data=str(e), usage=None, raw_response=completion)

    def track_completion_usage(self, completion: Any) -> CompletionUsage:
        usage: GroqCompletionUsage = completion.usage
        return CompletionUsage(
            completion_tokens=usage.completion_tokens,
            prompt_tokens=usage.prompt_tokens,
            total_tokens=usage.total_tokens,
        )

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.LLM:
            messages: Sequence[Dict[str, str]] = []
            if input is not None and input != "":
                messages.append({"role": "system", "content": input})
            final_model_kwargs["messages"] = messages
        else:
            raise ValueError(f"model_type {model_type} is not supported")
        return final_model_kwargs

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
        ),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        assert (
            "model" in api_kwargs
        ), f"model must be specified in api_kwargs: {api_kwargs}"
        if model_type == ModelType.LLM:
            completion = self.sync_client.chat.completions.create(**api_kwargs)
            return completion
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
        ),
        max_time=5,
    )
    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        if self.async_client is None:
            self.async_client = self.init_async_client()
        assert "model" in api_kwargs, "model must be specified"
        if model_type == ModelType.LLM:
            completion = await self.async_client.chat.completions.create(**api_kwargs)
            return completion
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @classmethod
    def from_dict(cls: type[T], data: Dict[str, Any]) -> T:
        obj = super().from_dict(data)
        # recreate the existing clients
        obj.sync_client = obj.init_sync_client()
        obj.async_client = obj.init_async_client()
        return obj

    def to_dict(self) -> Dict[str, Any]:
        r"""Convert the component to a dictionary."""
        # TODO: not exclude but save yes or no for recreating the clients
        exclude = [
            "sync_client",
            "async_client",
        ]  # unserializable object
        output = super().to_dict(exclude=exclude)
        return output
