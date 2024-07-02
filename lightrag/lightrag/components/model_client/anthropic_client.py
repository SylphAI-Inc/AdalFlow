"""Anthropic ModelClient integration."""

import os
from typing import Dict, Optional, Any


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


from lightrag.core.model_client import ModelClient
from lightrag.core.types import ModelType

from lightrag.utils.lazy_import import safe_import, OptionalPackages

safe_import(OptionalPackages.ANTHROPIC.value[0], OptionalPackages.ANTHROPIC.value[1])


class AnthropicAPIClient(ModelClient):
    __doc__ = r"""A component wrapper for the Anthropic API client.

    Visit https://docs.anthropic.com/en/docs/intro-to-claude for more api details.
    """

    def __init__(self, api_key: Optional[str] = None):
        r"""It is recommended to set the ANTHROPIC_API_KEY environment variable instead of passing it as an argument."""
        super().__init__()
        self._api_key = api_key
        self.sync_client = self.init_sync_client()
        self.async_client = None  # only initialize if the async call is called
        self.tested_llm_models = ["claude-3-opus-20240229"]

    def init_sync_client(self):
        api_key = self._api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Environment variable ANTHROPIC_API_KEY must be set")
        return anthropic.Anthropic(api_key=api_key)

    def init_async_client(self):
        api_key = self._api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Environment variable ANTHROPIC_API_KEY must be set")
        return anthropic.AsyncAnthropic(api_key=api_key)

    def parse_chat_completion(self, completion: Message) -> str:
        print(f"completion: {completion}")
        return completion.content[0].text

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> dict:
        api_kwargs = model_kwargs.copy()
        if model_type == ModelType.LLM:
            api_kwargs["messages"] = [
                {"role": "user", "content": input},
            ]
            if input and input != "":
                api_kwargs["system"] = input
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
            self.async_client = self.init_async_client()
        if model_type == ModelType.EMBEDDER:
            raise ValueError(f"Model type {model_type} not supported")
        elif model_type == ModelType.LLM:
            return await self.async_client.messages.create(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")
