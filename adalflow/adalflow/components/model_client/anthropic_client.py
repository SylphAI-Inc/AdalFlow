"""Anthropic ModelClient integration with OpenAI SDK compatibility."""

import os
import time
from typing import Dict, Optional, Any, Callable, Union
import backoff
import logging
from collections.abc import AsyncIterator, Iterator


from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    ModelType,
    CompletionUsage,
    GeneratorOutput,
    ResponseUsage,
    InputTokensDetails,
    OutputTokensDetails,
)

# optional import
from adalflow.utils.lazy_import import safe_import, OptionalPackages

# OpenAI SDK imports for Anthropic compatibility
openai = safe_import(OptionalPackages.OPENAI.value[0], OptionalPackages.OPENAI.value[1])

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai._streaming import Stream, AsyncStream
from openai import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
    BadRequestError,
)

# Import our ChatCompletion to Response converter utility
from adalflow.components.model_client.chat_completion_to_response_converter import (
    ChatCompletionToResponseConverter,
)
from openai.types.responses import Response

# Legacy Anthropic imports (commented out for reference)
# anthropic = safe_import(
#     OptionalPackages.ANTHROPIC.value[0], OptionalPackages.ANTHROPIC.value[1]
# )
# from anthropic import (
#     RateLimitError,
#     APITimeoutError,
#     InternalServerError,
#     UnprocessableEntityError,
#     BadRequestError,
#     MessageStreamManager,
# )
# from anthropic.types import Message, Usage, Completion

log = logging.getLogger(__name__)

# Legacy LLMResponse and parsers (commented out for reference)
# class LLMResponse(BaseModel):
#     """
#     LLM Response
#     """
#     content: Optional[str] = None
#     thinking: Optional[str] = None
#     tool_use: Optional[Function] = None

# def get_first_message_content(completion: Message) -> LLMResponse:
#     r"""When we only need the content of the first message.
#     It is the default parser for chat completion."""
#     try:
#         first_message = completion.content
#     except Exception as e:
#         log.error(f"Error getting the first message: {e}")
#         return LLMResponse()
#     output = LLMResponse()
#     for block in first_message:
#         if block.type == "text":
#             output.content = block.text
#         elif block.type == "thinking":
#             output.thinking = block.thinking
#         elif block.type == "tool_use":
#             name = block.name
#             input = block.input
#             output.tool_use = Function(name=name, kwargs=input)
#     return output

# def handle_streaming_response(generator: MessageStreamManager):
#     r"""Handle the streaming response."""
#     stream = generator.__enter__()
#     log.debug(f"stream: {stream}")
#     try:
#         for chunk in stream:
#             yield chunk
#     finally:
#         stream.__exit__()


def get_chat_completion_usage(completion: ChatCompletion) -> ResponseUsage:
    """Convert ChatCompletion.usage into our AdalFlowResponseUsage format."""
    usage = getattr(completion, "usage", None)

    # No usage present: return all zeros
    if not usage:
        return ResponseUsage(
            input_tokens=0,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=0,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=0,
        )

    # Extract nested details safely
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)

    input_tokens_details = InputTokensDetails(
        cached_tokens=getattr(prompt_details, "cached_tokens", 0)
    )
    output_tokens_details = OutputTokensDetails(
        reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0)
    )

    return ResponseUsage(
        input_tokens=getattr(usage, "prompt_tokens", 0),
        input_tokens_details=input_tokens_details,
        output_tokens=getattr(usage, "completion_tokens", 0),
        output_tokens_details=output_tokens_details,
        total_tokens=getattr(usage, "total_tokens", 0),
    )


__all__ = ["AnthropicAPIClient"]

FAKE_RESPONSES_ID = "fake_responses_id"

"""

# previous implementation
# class AnthropicAPIClient(ModelClient):
#     __doc__ = r"A component wrapper for the Anthropic API client.

#     Visit https://docs.anthropic.com/en/docs/intro-to-claude for more api details.

#     Note:

#     As antropic API needs users to set max_tokens, we set up a default value of 512 for the max_tokens.
#     You can override this value by passing the max_tokens in the model_kwargs.

#     Reference:
#     - https://docs.anthropic.com/en/docs/about-claude/models
#     - interlevad thinking:https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#interleaved-thinking
#     """

#     def __init__(
#         self,
#         api_key: Optional[str] = None,
#         non_streaming_chat_completion_parser: Callable[[Completion], Any] = None,
#         streaming_chat_completion_parser: Callable[[Completion], Any] = None,
#     ):
#         r"""It is recommended to set the ANTHROPIC_API_KEY environment variable instead of passing it as an argument."""
#         super().__init__()
#         self._api_key = api_key
#         self.sync_client = self.init_sync_client()
#         self.async_client = None  # only initialize if the async call is called
#         self.tested_llm_models = ["claude-3-opus-20240229", "claude-sonnet-4-20250514"]
#         self.non_streaming_chat_completion_parser = (
#             non_streaming_chat_completion_parser or get_first_message_content
#         )
#         self.streaming_chat_completion_parser = (
#             streaming_chat_completion_parser or handle_streaming_response
#         )
#         self.chat_completion_parser = self.non_streaming_chat_completion_parser
#         self.default_max_tokens = 512

#     def init_sync_client(self):
#         api_key = self._api_key or os.getenv("ANTHROPIC_API_KEY")
#         if not api_key:
#             raise ValueError("Environment variable ANTHROPIC_API_KEY must be set")
#         return anthropic.Anthropic(api_key=api_key)

#     def init_async_client(self):
#         api_key = self._api_key or os.getenv("ANTHROPIC_API_KEY")
#         if not api_key:
#             raise ValueError("Environment variable ANTHROPIC_API_KEY must be set")
#         return anthropic.AsyncAnthropic(api_key=api_key)

#     def parse_chat_completion(
#         self,
#         completion: Union[Completion, GeneratorType[MessageStreamManager, None, None]],
#     ) -> "GeneratorOutput":
#         """Parse the completion, and put it into the raw_response."""
#         log.debug(f"completion: {completion}, parser: {self.chat_completion_parser}")
#         try:
#             data = self.chat_completion_parser(completion)
#         except Exception as e:
#             log.error(f"Error parsing the completion: {e}")
#             return GeneratorOutput(data=None, error=str(e), raw_response=completion)

#         try:
#             usage = self.track_completion_usage(completion)
#             if isinstance(data, LLMResponse):
#                 return GeneratorOutput(
#                     data=None,
#                     error=None,
#                     raw_response=data.content,  # the final text answer
#                     thinking=data.thinking,
#                     tool_use=data.tool_use,
#                     usage=usage,
#                 )
#             else:
#                 # data will be the one parsed from the raw response
#                 return GeneratorOutput(
#                     data=None, error=None, raw_response=data, usage=usage
#                 )
#         except Exception as e:
#             log.error(f"Error tracking the completion usage: {e}")
#             return GeneratorOutput(data=None, error=str(e), raw_response=data)

#     def track_completion_usage(self, completion: Message) -> CompletionUsage:
#         r"""Track the completion usage."""
#         usage: Usage = completion.usage
#         return CompletionUsage(
#             completion_tokens=usage.output_tokens,
#             prompt_tokens=usage.input_tokens,
#             total_tokens=usage.output_tokens + usage.input_tokens,
#         )

#     # TODO: potentially use <SYS></SYS> to separate the system and user messages. This requires user to follow it. If it is not found, then we will only use user message.
#     def convert_inputs_to_api_kwargs(
#         self,
#         input: Optional[Any] = None,
#         model_kwargs: Dict = {},
#         model_type: ModelType = ModelType.UNDEFINED,
#     ) -> dict:
#         r"""Anthropic API messages separates the system and the user messages.

#         As we focus on one prompt, we have to use the user message as the input.

#         api: https://docs.anthropic.com/en/api/messages
#         """
#         api_kwargs = model_kwargs.copy()
#         if model_type == ModelType.LLM:
#             api_kwargs["messages"] = [
#                 {"role": "user", "content": input},
#             ]
#             if "max_tokens" not in api_kwargs:
#                 api_kwargs["max_tokens"] = self.default_max_tokens
#             # if input and input != "":
#             #     api_kwargs["system"] = input
#         else:
#             raise ValueError(f"Model type {model_type} not supported")
#         return api_kwargs

#     @backoff.on_exception(
#         backoff.expo,
#         (
#             APITimeoutError,
#             InternalServerError,
#             RateLimitError,
#             UnprocessableEntityError,
#             BadRequestError,
#         ),
#         max_time=5,
#     )
#     def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
#         """
#         kwargs is the combined input and model_kwargs
#         """
#         if model_type == ModelType.EMBEDDER:
#             raise ValueError(f"Model type {model_type} not supported")
#         elif model_type == ModelType.LLM:
#             if "stream" in api_kwargs and api_kwargs.get("stream", False):
#                 log.debug("streaming call")
#                 # remove the stream from the api_kwargs
#                 api_kwargs.pop("stream", None)
#                 self.chat_completion_parser = self.streaming_chat_completion_parser
#                 return self.sync_client.messages.stream(**api_kwargs)
#             else:
#                 log.debug("non-streaming call")
#                 self.chat_completion_parser = self.non_streaming_chat_completion_parser
#                 return self.sync_client.messages.create(**api_kwargs)
#         else:
#             raise ValueError(f"model_type {model_type} is not supported")

#     @backoff.on_exception(
#         backoff.expo,
#         (
#             APITimeoutError,
#             InternalServerError,
#             RateLimitError,
#             UnprocessableEntityError,
#             BadRequestError,
#         ),
#         max_time=5,
#     )
#     async def acall(
#         self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
#     ):
#         """
#         kwargs is the combined input and model_kwargs
#         """
#         if self.async_client is None:
#             self.async_client = self.init_async_client()
#         if model_type == ModelType.EMBEDDER:
#             raise ValueError(f"Model type {model_type} not supported")
#         elif model_type == ModelType.LLM:
#             return await self.async_client.messages.create(**api_kwargs)
#         else:
#             raise ValueError(f"model_type {model_type} is not supported")


class AnthropicAPIClient(ModelClient):
    """A component wrapper for Anthropic API using OpenAI SDK compatibility.

    This client leverages Anthropic's OpenAI SDK compatibility layer to provide
    ChatCompletion-based responses while maintaining AdalFlow's GeneratorOutput structure.

    Features:
    - Uses OpenAI SDK with Anthropic's compatibility endpoint
    - Supports both streaming and non-streaming calls
    - Handles ModelType.LLM and ModelType.LLM_REASONING
    - Converts ChatCompletion responses to Response API format for compatibility
    - Maintains backward compatibility with existing AdalFlow parsers

    Args:
        api_key (Optional[str]): Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
        base_url (str): Anthropic's OpenAI compatibility endpoint.
        non_streaming_chat_completion_parser (Callable): Legacy parser for non-streaming
            ChatCompletion objects. Used for backward compatibility with existing code
            that depends on the original AnthropicAPI client's parsing behavior.
        streaming_chat_completion_parser (Callable): Parser for streaming ChatCompletion
            responses. Handles conversion from ChatCompletion streams to Response API format.

    Note:
        Requires ANTHROPIC_API_KEY environment variable or api_key parameter.
        Uses OpenAI SDK internally but calls Anthropic's API via compatibility layer.
        The non_streaming_chat_completion_parser is provided for legacy compatibility only.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com/v1/",
        non_streaming_chat_completion_parser: Optional[
            Callable[[ChatCompletion], Any]
        ] = None,
        streaming_chat_completion_parser: Optional[Callable[[AsyncStream], Any]] = None,
    ):
        """Initialize Anthropic client with OpenAI SDK compatibility."""
        super().__init__()
        self._api_key = api_key
        self.base_url = base_url
        self.sync_client = self.init_sync_client()
        self.async_client = None  # Initialize lazily
        self._api_kwargs = {}  # Store API kwargs for tracing

        # Set up parsers - using ChatCompletion format
        # Legacy parser for backward compatibility with existing AnthropicAPI client code
        self.non_streaming_chat_completion_parser = (
            non_streaming_chat_completion_parser
            or ChatCompletionToResponseConverter.get_chat_completion_content
        )
        self.streaming_chat_completion_parser = (
            streaming_chat_completion_parser
            or ChatCompletionToResponseConverter.sync_handle_stream
        )

        # Default model parameters
        self.default_max_tokens = 4096

        # Tested models
        # self.tested_llm_models = ["claude-3-opus-20240229", "claude-sonnet-4-20250514"]

    def init_sync_client(self):
        """Initialize synchronous OpenAI client pointing to Anthropic."""
        api_key = self._api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Environment variable ANTHROPIC_API_KEY must be set")

        return OpenAI(api_key=api_key, base_url=self.base_url)

    def init_async_client(self):
        """Initialize asynchronous OpenAI client pointing to Anthropic."""
        api_key = self._api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Environment variable ANTHROPIC_API_KEY must be set")

        return AsyncOpenAI(api_key=api_key, base_url=self.base_url)

    def parse_chat_completion(
        self,
        completion: Union[
            ChatCompletion,
            Stream[ChatCompletionChunk],
            AsyncStream[ChatCompletionChunk],
        ],
    ) -> GeneratorOutput:
        """
        Parse ChatCompletion and convert to GeneratorOutput with Response API compatibility.

        This method uses ChatCompletionToResponseConverter to transform ChatCompletion
        objects into text format compatible with existing Response API parsers.

        Args:
            completion: ChatCompletion object or stream from OpenAI SDK

        Returns:
            GeneratorOutput with converted raw_response text
        """
        log.debug(f"Parsing completion type: {type(completion)}")

        output = None
        usage = None  # Initialize usage to None

        # TODO add other fields such as model and tool_choice that can be passed in from the Generator
        # from OpenAI Agent SDK under OpenAIChatCompletionsModel and fields that are not used for response.created is commented out
        # follows documentation of https://platform.openai.com/docs/api-reference/realtime-server-events/response

        # response = Response(
        #     id=FAKE_RESPONSES_ID,
        #     created_at=time.time(),
        #     model=self.model,
        #     object="response",
        #     output=[],
        #     tool_choice=cast(Literal["auto", "required", "none"], tool_choice)
        #     if tool_choice != NOT_GIVEN
        #     else "auto",
        #     top_p=model_settings.top_p,
        #     temperature=model_settings.temperature,
        #     tools=[],
        #     parallel_tool_calls=parallel_tool_calls or False,
        #     reasoning=model_settings.reasoning,
        # )

        response = Response(
            id=FAKE_RESPONSES_ID,
            created_at=time.time(),
            model="anthropic",  # specify and need to replace
            object="response",
            output=[],
            tool_choice="auto",  # the model is free to decide whether to use tools or not
            # tool_choice=cast(Literal["auto", "required", "none"], tool_choice)
            # if tool_choice != NOT_GIVEN
            # else "auto",
            # top_p=model_settings.top_p,
            # temperature=model_settings.temperature,
            tools=[],
            parallel_tool_calls=True,  # enabled by default for chat completions as well
            # reasoning=model_settings.reasoning,
        )

        try:
            if isinstance(completion, ChatCompletion):
                log.debug("Converting non-streaming ChatCompletion to text")
                # Non-streaming ChatCompletion - convert to text
                output = ChatCompletionToResponseConverter.get_chat_completion_content(
                    completion
                )
                usage = get_chat_completion_usage(completion)

            elif isinstance(completion, AsyncIterator):
                # Async iterable case - convert using ChatCompletionToResponseConverter
                log.debug(
                    "Converting async iterable stream to Response API compatible events"
                )
                output = ChatCompletionToResponseConverter.async_handle_stream(
                    response, completion
                )

            elif isinstance(completion, Iterator):
                # Sync iterable case - convert using ChatCompletionToResponseConverter
                log.debug(
                    "Converting sync iterable stream to Response API compatible events"
                )
                output = ChatCompletionToResponseConverter.sync_handle_stream(
                    response, completion
                )

            else:
                # Fallback for other types
                log.warning(f"Unexpected completion type: {type(completion)}")

            return GeneratorOutput(
                data=None,
                api_response=completion,  # return the original completion
                raw_response=output,  # store the original output in raw response
                usage=usage,
                error=None,
            )

        except Exception as e:
            log.error(f"Error parsing ChatCompletion: {e}")
            return GeneratorOutput(
                data=None,
                api_response=completion,
                raw_response="",
                usage=None,
                error=str(e),
            )

    def track_completion_usage(self, completion: ChatCompletion) -> CompletionUsage:
        """Track completion usage from ChatCompletion object."""
        if not completion.usage:
            return CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)

        return CompletionUsage(
            completion_tokens=completion.usage.completion_tokens,
            prompt_tokens=completion.usage.prompt_tokens,
            total_tokens=completion.usage.total_tokens,
        )

    # slightly modified from the original AnthropicAPIClient's convert_inputs_to_api_kwargs function
    # which was based on the Anthropic's Message API https://docs.anthropic.com/en/api/messages
    # notably OpenAI's message API requires placing system under the role field
    # but Anthropic's Message API has a system as a top level field
    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """
        Convert AdalFlow inputs to OpenAI ChatCompletion API format.

        Converts single input text to OpenAI messages format expected by
        chat.completions.create endpoint.

        Args:
            input: Text input or messages array
            model_kwargs: Additional model parameters
            model_type: Type of model (LLM or LLM_REASONING)

        Returns:
            Dict: API kwargs formatted for OpenAI chat.completions.create

        convertible with original api of Anthropic's Message API
        """
        api_kwargs = model_kwargs.copy()

        if model_type in [ModelType.LLM, ModelType.LLM_REASONING]:
            # Convert input to messages format
            if isinstance(input, str):
                api_kwargs["messages"] = [{"role": "user", "content": input}]
            elif isinstance(input, list):
                # Assume it's already in messages format
                api_kwargs["messages"] = input
            else:
                api_kwargs["messages"] = [{"role": "user", "content": str(input)}]

            # Set default max_tokens if not provided
            if "max_tokens" not in api_kwargs:
                api_kwargs["max_tokens"] = self.default_max_tokens

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
        Synchronous call to Anthropic via OpenAI SDK compatibility.

        Supports both LLM and LLM_REASONING model types with streaming and non-streaming.

        Args:
            api_kwargs: API parameters for chat.completions.create
            model_type: ModelType.LLM or ModelType.LLM_REASONING

        Returns:
            ChatCompletion or Stream[ChatCompletionChunk] from Anthropic
        """
        log.info(f"Anthropic API call with model_type: {model_type}")
        self._api_kwargs = api_kwargs

        if model_type == ModelType.EMBEDDER:
            raise ValueError(f"Model type {model_type} not supported")

        elif model_type in [ModelType.LLM, ModelType.LLM_REASONING]:
            if api_kwargs.get("stream", False):
                log.debug("Streaming call via OpenAI SDK to Anthropic")
                return self.sync_client.chat.completions.create(**api_kwargs)
            else:
                log.debug("Non-streaming call via OpenAI SDK to Anthropic")
                return self.sync_client.chat.completions.create(**api_kwargs)

        else:
            raise ValueError(f"Model type {model_type} is not supported")

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
        Asynchronous call to Anthropic via OpenAI SDK compatibility.

        Args:
            api_kwargs: API parameters for chat.completions.create
            model_type: ModelType.LLM or ModelType.LLM_REASONING

        Returns:
            ChatCompletion or AsyncStream[ChatCompletionChunk] from Anthropic
        """
        self._api_kwargs = api_kwargs

        if self.async_client is None:
            self.async_client = self.init_async_client()

        if model_type == ModelType.EMBEDDER:
            raise ValueError(f"Model type {model_type} not supported")

        elif model_type in [ModelType.LLM, ModelType.LLM_REASONING]:
            log.debug(
                f"Async call via OpenAI SDK to Anthropic, streaming: {api_kwargs.get('stream', False)}"
            )
            return await self.async_client.chat.completions.create(**api_kwargs)

        else:
            raise ValueError(f"Model type {model_type} is not supported")
