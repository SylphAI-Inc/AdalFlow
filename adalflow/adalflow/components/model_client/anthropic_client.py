"""Anthropic ModelClient integration."""

import os
from typing import Dict, Optional, Any, Callable
import backoff
import logging


from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, CompletionUsage, GeneratorOutput

# optional import
from adalflow.utils.lazy_import import safe_import, OptionalPackages

anthropic = safe_import(
    OptionalPackages.ANTHROPIC.value[0], OptionalPackages.ANTHROPIC.value[1]
)
import anthropic
from anthropic import (
    RateLimitError,
    APITimeoutError,
    InternalServerError,
    UnprocessableEntityError,
    BadRequestError,
)
from anthropic.types import Message, Usage

log = logging.getLogger(__name__)


def get_first_message_content(completion: Message) -> str:
    r"""When we only need the content of the first message.
    It is the default parser for chat completion."""
    return completion.content[0].text


__all__ = ["AnthropicAPIClient", "get_first_message_content"]


# NOTE: using customize parser might make the new_component more complex when we have to handle a callable
class AnthropicAPIClient(ModelClient):
    __doc__ = r"""A component wrapper for the Anthropic API client.

    Visit https://docs.anthropic.com/en/docs/intro-to-claude for more api details.

    Ensure "max_tokens" are set.

    Reference: 8/1/2024
    - https://docs.anthropic.com/en/docs/about-claude/models
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Message], Any] = None,
    ):
        r"""It is recommended to set the ANTHROPIC_API_KEY environment variable instead of passing it as an argument."""
        super().__init__()
        self._api_key = api_key
        self.sync_client = self.init_sync_client()
        self.async_client = None  # only initialize if the async call is called
        self.tested_llm_models = ["claude-3-opus-20240229"]
        self.chat_completion_parser = (
            chat_completion_parser or get_first_message_content
        )

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

    def parse_chat_completion(self, completion: Message) -> GeneratorOutput:
        log.debug(f"completion: {completion}")
        try:
            data = completion.content[0].text
            usage = self.track_completion_usage(completion)
            return GeneratorOutput(data=None, usage=usage, raw_response=data)
        except Exception as e:
            log.error(f"Error parsing completion: {e}")
            return GeneratorOutput(
                data=None, error=str(e), raw_response=str(completion)
            )

    def track_completion_usage(self, completion: Message) -> CompletionUsage:
        r"""Track the completion usage."""
        usage: Usage = completion.usage
        return CompletionUsage(
            completion_tokens=usage.output_tokens,
            prompt_tokens=usage.input_tokens,
            total_tokens=usage.output_tokens + usage.input_tokens,
        )

    # TODO: potentially use <SYS></SYS> to separate the system and user messages. This requires user to follow it. If it is not found, then we will only use user message.
    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> dict:
        r"""Anthropic API messages separates the system and the user messages.

        As we focus on one prompt, we have to use the user message as the input.

        api: https://docs.anthropic.com/en/api/messages
        """
        api_kwargs = model_kwargs.copy()
        if model_type == ModelType.LLM:
            api_kwargs["messages"] = [
                {"role": "user", "content": input},
            ]
            # if input and input != "":
            #     api_kwargs["system"] = input
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
