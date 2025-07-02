"""Anthropic ModelClient integration."""

import os
from typing import Dict, Optional, Any, Callable, Union, Generator as GeneratorType
import backoff
import logging


from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, CompletionUsage, GeneratorOutput, Function

# optional import
from adalflow.utils.lazy_import import safe_import, OptionalPackages

anthropic = safe_import(
    OptionalPackages.ANTHROPIC.value[0], OptionalPackages.ANTHROPIC.value[1]
)

# import anthropic
from anthropic import (
    RateLimitError,
    APITimeoutError,
    InternalServerError,
    UnprocessableEntityError,
    BadRequestError,
    MessageStreamManager,
)
from anthropic.types import Message, Usage, Completion
from pydantic import BaseModel

log = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """
    LLM Response
    """

    content: Optional[str] = None
    thinking: Optional[str] = None
    tool_use: Optional[Function] = None


def get_first_message_content(completion: Message) -> LLMResponse:
    r"""When we only need the content of the first message.
    It is the default parser for chat completion."""
    try:
        first_message = completion.content
    except Exception as e:
        log.error(f"Error getting the first message: {e}")
        return LLMResponse()

    output = LLMResponse()
    for block in first_message:
        if block.type == "text":
            output.content = block.text
        elif block.type == "thinking":
            output.thinking = block.thinking
        elif block.type == "tool_use":
            name = block.name
            input = block.input
            output.tool_use = Function(name=name, kwargs=input)

    return output


def handle_streaming_response(generator: MessageStreamManager):
    r"""Handle the streaming response."""
    stream = generator.__enter__()
    log.debug(f"stream: {stream}")
    try:
        for chunk in stream:
            yield chunk
    finally:
        stream.__exit__()


__all__ = ["AnthropicAPIClient", "get_first_message_content"]


# NOTE: using customize parser might make the new_component more complex when we have to handle a callable
class AnthropicAPIClient(ModelClient):
    __doc__ = r"""A component wrapper for the Anthropic API client.

    Visit https://docs.anthropic.com/en/docs/intro-to-claude for more api details.

    Note:

    As antropic API needs users to set max_tokens, we set up a default value of 512 for the max_tokens.
    You can override this value by passing the max_tokens in the model_kwargs.

    Reference:
    - https://docs.anthropic.com/en/docs/about-claude/models
    - interlevad thinking:https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#interleaved-thinking
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        non_streaming_chat_completion_parser: Callable[[Completion], Any] = None,
        streaming_chat_completion_parser: Callable[[Completion], Any] = None,
    ):
        r"""It is recommended to set the ANTHROPIC_API_KEY environment variable instead of passing it as an argument."""
        super().__init__()
        self._api_key = api_key
        self.sync_client = self.init_sync_client()
        self.async_client = None  # only initialize if the async call is called
        self.tested_llm_models = ["claude-3-opus-20240229", "claude-sonnet-4-20250514"]
        self.non_streaming_chat_completion_parser = (
            non_streaming_chat_completion_parser or get_first_message_content
        )
        self.streaming_chat_completion_parser = (
            streaming_chat_completion_parser or handle_streaming_response
        )
        self.chat_completion_parser = self.non_streaming_chat_completion_parser
        self.default_max_tokens = 512

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

    def parse_chat_completion(
        self,
        completion: Union[Completion, GeneratorType[MessageStreamManager, None, None]],
    ) -> "GeneratorOutput":
        """Parse the completion, and put it into the raw_response."""
        log.debug(f"completion: {completion}, parser: {self.chat_completion_parser}")
        try:
            data = self.chat_completion_parser(completion)
        except Exception as e:
            log.error(f"Error parsing the completion: {e}")
            return GeneratorOutput(data=None, error=str(e), raw_response=completion)

        try:
            usage = self.track_completion_usage(completion)
            if isinstance(data, LLMResponse):
                return GeneratorOutput(
                    data=None,
                    error=None,
                    raw_response=data.content,  # the final text answer
                    thinking=data.thinking,
                    tool_use=data.tool_use,
                    usage=usage,
                )
            else:
                # data will be the one parsed from the raw response
                return GeneratorOutput(
                    data=None, error=None, raw_response=data, usage=usage
                )
        except Exception as e:
            log.error(f"Error tracking the completion usage: {e}")
            return GeneratorOutput(data=None, error=str(e), raw_response=data)

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
            if "max_tokens" not in api_kwargs:
                api_kwargs["max_tokens"] = self.default_max_tokens
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
            if "stream" in api_kwargs and api_kwargs.get("stream", False):
                log.debug("streaming call")
                # remove the stream from the api_kwargs
                api_kwargs.pop("stream", None)
                self.chat_completion_parser = self.streaming_chat_completion_parser
                return self.sync_client.messages.stream(**api_kwargs)
            else:
                log.debug("non-streaming call")
                self.chat_completion_parser = self.non_streaming_chat_completion_parser
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
