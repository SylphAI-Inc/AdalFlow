# https://docs.anthropic.com/en/api/messages
import os
from typing import Any, Dict, Optional, Sequence, Union
import anthropic
from anthropic import (
    RateLimitError,
    APITimeoutError,
    InternalServerError,
    UnprocessableEntityError,
    BadRequestError,
)
from anthropic.types import Message
import backoff


from core.api_client import APIClient
from core.data_classes import ModelType


class AnthropicAPIClient(APIClient):
    __doc__ = r"""A component wrapper for the Anthropic API client.

    Visit https://docs.anthropic.com/en/docs/intro-to-claude for more api details.
    """

    def __init__(self, api_key: Optional[str] = None):
        r"""It is recommended to set the ANTHROPIC_API_KEY environment variable instead of passing it as an argument."""
        super().__init__()
        self._api_key = api_key
        self.sync_client = self._init_sync_client()
        self.async_client = None  # only initialize if the async call is called

    def _init_sync_client(self):
        api_key = self._api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Environment variable ANTHROPIC_API_KEY must be set")
        return anthropic.Anthropic(api_key=api_key)

    def _init_async_client(self):
        api_key = self._api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Environment variable ANTHROPIC_API_KEY must be set")
        return anthropic.AsyncAnthropic(api_key=api_key)

    def parse_chat_completion(self, completion: Message) -> str:
        print(f"completion: {completion}")
        return completion.content[0].text

    def convert_input_to_api_kwargs(
        self,
        input: Union[str, Sequence],
        system_input: Optional[Union[str]] = None,
        combined_model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> dict:
        api_kwargs = combined_model_kwargs.copy()
        if model_type == ModelType.LLM:
            api_kwargs["messages"] = [
                {"role": "user", "content": input},
            ]
            if system_input and system_input != "":
                api_kwargs["system"] = system_input
        else:
            raise ValueError(f"Model type {model_type} not supported")
        return api_kwargs

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
            BadRequestError,
        ),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """
        kwargs is the combined input and model_kwargs
        """
        if model_type == ModelType.EMBEDDER:
            raise ValueError(f"Model type {model_type} not supported")
        elif model_type == ModelType.LLM:
            return self.sync_client.messages.create(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
            BadRequestError,
        ),
        max_time=5,
    )
    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        """
        kwargs is the combined input and model_kwargs
        """
        if self.async_client is None:
            self.async_client = self._init_async_client()
        if model_type == ModelType.EMBEDDER:
            raise ValueError(f"Model type {model_type} not supported")
        elif model_type == ModelType.LLM:
            return await self.async_client.messages.create(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")
