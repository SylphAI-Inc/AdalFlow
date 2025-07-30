"""OpenAI ModelClient integration."""

import os
import base64
from typing import (
    Dict,
    Sequence,
    Optional,
    List,
    Any,
    TypeVar,
    Callable,
    Generator as GeneratorType,
    Union,
    Literal,
    Iterable,
    AsyncIterable,
    AsyncGenerator,
)
import re

import logging
import backoff

# optional import
from adalflow.utils.lazy_import import safe_import, OptionalPackages


openai = safe_import(OptionalPackages.OPENAI.value[0], OptionalPackages.OPENAI.value[1])

from openai import (
    OpenAI,
    AsyncOpenAI,
)  # , Stream  # COMMENTED OUT - USING RESPONSE API ONLY
from openai import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
    BadRequestError,
)
from openai.types import (
    Completion,
    CreateEmbeddingResponse,
    Image,
)

# from openai.types.chat import ChatCompletionChunk, ChatCompletion  # COMMENTED OUT - USING RESPONSE API ONLY
from openai.types.responses import Response, ResponseUsage
from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    ModelType,
    EmbedderOutput,
    ResponseUsage as AdalFlowResponseUsage,
    InputTokensDetails,
    OutputTokensDetails,
    GeneratorOutput,
)

from adalflow.components.model_client.utils import parse_embedding_response

log = logging.getLogger(__name__)
T = TypeVar("T")


# OLD CHAT COMPLETION PARSING FUNCTIONS (COMMENTED OUT)
# # completion parsing functions and you can combine them into one single chat completion parser
# def get_first_message_content(completion: ChatCompletion) -> str:
#     r"""When we only need the content of the first message.
#     It is the default parser for chat completion."""
#     log.debug(f"raw completion: {completion}")
#     return completion.choices[0].message.content


def get_response_output_text(response: Response) -> str:
    """Used to extract the data field for the reasoning model"""
    log.debug(f"raw response: {response}")
    return response.output_text


# def _get_chat_completion_usage(completion: ChatCompletion) -> OpenAICompletionUsage:
#     return completion.usage


# A simple heuristic to estimate token count for estimating number of tokens in a Streaming response
def estimate_token_count(text: str) -> int:
    """
    Estimate the token count of a given text.

    Args:
        text (str): The text to estimate token count for.

    Returns:
        int: Estimated token count.
    """
    # Split the text into tokens using spaces as a simple heuristic
    tokens = text.split()

    # Return the number of tokens
    return len(tokens)


# OLD CHAT COMPLETION STREAMING FUNCTIONS (COMMENTED OUT)
# def parse_stream_chat_completion(completion: ChatCompletionChunk) -> str:
#     r"""Parse the completion chunks of the chat completion API."""
#     output = completion.choices[0].delta.content
#     if hasattr(completion, "citations"):
#         citations = completion.citations
#         return output, citations
#     return output


# def handle_streaming_chat_completion(generator: Stream[ChatCompletionChunk]):
#     r"""Handle the streaming completion."""
#     for completion in generator:
#         log.debug(f"Raw chunk completion: {completion}")
#         parsed_content = parse_stream_chat_completion(completion)
#         yield parsed_content


async def handle_streaming_response(
    stream: AsyncIterable[Any],
) -> AsyncGenerator[str, None]:
    """
    Async generator that processes a stream of SSE events from client.responses.create(..., stream=True).

    Args:
        stream: An async iterable of SSE events from the OpenAI API

    Yields:
        str: Non-empty text fragments parsed from the stream events
    """
    async for event in stream:
        yield event


def handle_streaming_response_sync(stream: Iterable) -> GeneratorType:
    """
    Synchronous version: Iterate over an SSE stream from client.responses.create(..., stream=True),
    logging each raw event and yielding non-empty text fragments.
    """
    # already compatible as this is the OpenAI client
    for event in stream:
        yield event


# OLD CHAT COMPLETION UTILITY FUNCTIONS (COMMENTED OUT)
# def get_all_messages_content(completion: ChatCompletion) -> List[str]:
#     r"""When the n > 1, get all the messages content."""
#     return [c.message.content for c in completion.choices]


# def get_probabilities(completion: ChatCompletion) -> List[List[TokenLogProb]]:
#     r"""Get the probabilities of each token in the completion."""
#     log_probs = []
#     for c in completion.choices:
#         content = c.logprobs.content
#         print(content)
#         log_probs_for_choice = []
#         for openai_token_logprob in content:
#             token = openai_token_logprob.token
#             logprob = openai_token_logprob.logprob
#             log_probs_for_choice.append(TokenLogProb(token=token, logprob=logprob))
#         log_probs.append(log_probs_for_choice)
#     return log_probs


class OpenAIClient(ModelClient):
    __doc__ = r"""A component wrapper for the OpenAI API client.

    Support both embedding and chat completion API, including multimodal capabilities.

    Users (1) simplify use ``Embedder`` and ``Generator`` components by passing OpenAIClient() as the model_client.
    (2) can use this as an example to create their own API client or extend this class(copying and modifing the code) in their own project.

    Note:
        We suggest users not to use `response_format` to enforce output data type or `tools` and `tool_choice`  in your model_kwargs when calling the API.
        We do not know how OpenAI is doing the formating or what prompt they have added.
        Instead
        - use :ref:`OutputParser<components-output_parsers>` for response parsing and formating.

        For multimodal inputs, provide images in model_kwargs["images"] as a path, URL, or list of them.
        The model must support vision capabilities (e.g., gpt-4o, gpt-4o-mini, o1, o1-mini).

        For image generation, use model_type=ModelType.IMAGE_GENERATION and provide:
        - model: "dall-e-3" or "dall-e-2"
        - prompt: Text description of the image to generate
        - size: "1024x1024", "1024x1792", or "1792x1024" for DALL-E 3; "256x256", "512x512", or "1024x1024" for DALL-E 2
        - quality: "standard" or "hd" (DALL-E 3 only)
        - n: Number of images to generate (1 for DALL-E 3, 1-10 for DALL-E 2)
        - response_format: "url" or "b64_json"

    Args:
        api_key (Optional[str], optional): OpenAI API key. Defaults to `None`.
        non_streaming_chat_completion_parser (Callable[[Completion], Any], optional): Legacy parser for chat completions.
            Defaults to `None` (deprecated).
        streaming_chat_completion_parser (Callable[[Completion], Any], optional): Legacy parser for streaming chat completions.
            Defaults to `None` (deprecated).
        non_streaming_response_parser (Callable[[Response], Any], optional): The parser for non-streaming responses.
            Defaults to `get_response_output_text`.
        streaming_response_parser (Callable[[Response], Any], optional): The parser for streaming responses.
            Defaults to `handle_streaming_response`.
        input_type (Literal["text", "messages"]): Input type for the client. Defaults to "text".
        base_url (str): The API base URL to use when initializing the client.
            Defaults to `"https://api.openai.com/v1/"`, but can be customized for third-party API providers or self-hosted models.
        env_api_key_name (str): The environment variable name for the API key. Defaults to `"OPENAI_API_KEY"`.
        organization (Optional[str], optional): OpenAI organization key. Defaults to None.
        headers (Optional[Dict[str, str]], optional): Additional headers to include in API requests. Defaults to None.

    References:
        - OpenAI API Overview: https://platform.openai.com/docs/introduction
        - Embeddings Guide: https://platform.openai.com/docs/guides/embeddings
        - Chat Completion Models: https://platform.openai.com/docs/guides/text-generation
        - Vision Models: https://platform.openai.com/docs/guides/vision
        - Image Generation: https://platform.openai.com/docs/guides/images

    Note:
        - Ensure each OpenAIClient instance is used by one generator only.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        # OLD CHAT COMPLETION PARSER PARAMS (kept for backward compatibility)
        non_streaming_chat_completion_parser: Optional[
            Callable[[Completion], Any]
        ] = None,  # non-streaming parser
        streaming_chat_completion_parser: Optional[
            Callable[[Completion], Any]
        ] = None,  # streaming parser
        # Response API parsers (used for reasoning models)
        non_streaming_response_parser: Optional[Callable[[Response], Any]] = None,
        streaming_response_parser: Optional[Callable[[Response], Any]] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: str = "https://api.openai.com/v1/",
        env_api_key_name: str = "OPENAI_API_KEY",
        organization: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        r"""It is recommended to set the OPENAI_API_KEY environment variable instead of passing it as an argument.

        Args:
            api_key (Optional[str], optional): OpenAI API key. Defaults to None.
            non_streaming_chat_completion_parser (Optional[Callable[[Completion], Any]], optional): Legacy parser for chat completions. Defaults to None.
            streaming_chat_completion_parser (Optional[Callable[[Completion], Any]], optional): Legacy parser for streaming chat completions. Defaults to None.
            non_streaming_response_parser (Optional[Callable[[Response], Any]], optional): Parser for non-streaming responses. Defaults to None.
            streaming_response_parser (Optional[Callable[[Response], Any]], optional): Parser for streaming responses. Defaults to None.
            input_type (Literal["text", "messages"]): Input type for the client. Defaults to "text".
            base_url (str): The API base URL to use when initializing the client.
            env_api_key_name (str): The environment variable name for the API key. Defaults to `"OPENAI_API_KEY"`.
            organization (Optional[str], optional): OpenAI organization key. Defaults to None.
            headers (Optional[Dict[str, str]], optional): Additional headers to include in API requests. Defaults to None.
        """
        super().__init__()
        self._api_key = api_key
        self.base_url = base_url
        self._env_api_key_name = env_api_key_name
        self.organization = organization
        self.headers = headers or {}
        self.sync_client = self.init_sync_client()
        self.async_client = None  # only initialize if the async call is called
        self._input_type = input_type
        self._api_kwargs = {}  # add api kwargs when the OpenAI Client is called

        # OLD CHAT COMPLETION API PARSERS (COMMENTED OUT)
        # # Chat Completion API Parsers
        # # (used only for synchronous (stream + non-streaming) calls via create API)
        # self.non_streaming_chat_completion_parser = (
        #     non_streaming_chat_completion_parser or get_first_message_content
        # )
        # self.streaming_chat_completion_parser = (
        #     streaming_chat_completion_parser or handle_streaming_chat_completion
        # )

        # Response API parsers (RESPONSE API ONLY NOW)
        # (used for both synchronous and asynchronous (stream + non-streaming) calls via Response API)
        self.non_streaming_response_parser = (
            non_streaming_response_parser or get_response_output_text
        )
        # Separate sync and async streaming parsers
        self.streaming_response_parser_sync = handle_streaming_response_sync
        self.streaming_response_parser_async = (
            streaming_response_parser or handle_streaming_response
        )

        # Default parsers (will be set dynamically based on sync/async context)
        self.response_parser = self.non_streaming_response_parser
        self.streaming_response_parser = (
            self.streaming_response_parser_async
        )  # Default to async
        # self.chat_completion_parser = self.non_streaming_chat_completion_parser  # COMMENTED OUT

    def init_sync_client(self):
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        if not api_key:
            raise ValueError(
                f"Environment variable {self._env_api_key_name} must be set"
            )
        return OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            organization=self.organization,
            default_headers=self.headers,
        )

    def init_async_client(self):
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        if not api_key:
            raise ValueError(
                f"Environment variable {self._env_api_key_name} must be set"
            )
        return AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            organization=self.organization,
            default_headers=self.headers,
        )

    # def _parse_chat_completion(self, completion: ChatCompletion) -> "GeneratorOutput":
    #     # TODO: raw output it is better to save the whole completion as a source of truth instead of just the message
    #     try:
    #         data = self.chat_completion_parser(completion)
    #         usage = self.track_completion_usage(completion)
    #         return GeneratorOutput(
    #             data=data, error=None, raw_response=str(data), usage=usage
    #         )
    #     except Exception as e:
    #         log.error(f"Error parsing the completion: {e}")
    #         return GeneratorOutput(data=None, error=str(e), raw_response=completion)

    # OLD CHAT COMPLETIONS API FUNCTION (COMMENTED OUT)
    # def parse_chat_completion(
    #     self,
    #     completion: Union[
    #         ChatCompletion, GeneratorType[ChatCompletionChunk, None, None], Response
    #     ],
    # ) -> "GeneratorOutput":
    #     """Function handles a lot of logic is used for parsing both stream responses and nonstream responses.
    #     # Determine parser based on completion type and streaming mode
    #     parser = (
    #         self.chat_completion_parser
    #         if isinstance(completion, ChatCompletion)
    #         else self.response_parser
    #     )
    #     log.debug(f"completion/response: {completion}, parser: {parser}")
    #     try:
    #         data = parser(completion)
    #     except Exception as e:
    #         log.error(f"Error parsing the completion: {e}")
    #         return GeneratorOutput(data=None, error=str(e), raw_response=completion)

    #     try:
    #         usage = self.track_completion_usage(completion)
    #         return GeneratorOutput(
    #             data=None, error=None, raw_response=data, usage=usage
    #         )
    #     except Exception as e:
    #         log.error(f"Error tracking the completion usage: {e}")
    #         return GeneratorOutput(data=None, error=str(e), raw_response=data)

    # NEW RESPONSE API ONLY FUNCTION
    def parse_chat_completion(
        self,
        completion: Union[Response, AsyncIterable],
    ) -> "GeneratorOutput":
        """Parse the Response API completion and put it into the raw_response.
        Fully migrated to Response API only."""

        parser = self.response_parser
        log.info(f"completion/response: {completion}, parser: {parser}")

        data = parser(completion)
        usage = self.track_completion_usage(completion)
        return GeneratorOutput(data=None, error=None, raw_response=data, usage=usage)

    # OLD CHAT COMPLETIONS API FUNCTION (COMMENTED OUT)
    # # NOTE: this is adapted to parse both completion and response
    # def track_completion_usage(
    #     self,
    #     completion: Union[
    #         ChatCompletion, GeneratorType[ChatCompletionChunk, None, None], Response, AsyncIterable
    #     ],
    # ) -> Union[CompletionUsage, ResponseUsage]:
    #     # Handle the case where completion is an async generator of response stream events
    #     try:
    #         if isinstance(completion, Response):
    #             # Handle Response object with ResponseUsage structure
    #             input_tokens_details = InputTokensDetails(
    #                 cached_tokens=getattr(completion.usage, "cached_tokens", 0)
    #             )

    #             output_tokens_details = OutputTokensDetails(
    #                 reasoning_tokens=getattr(completion.usage, "reasoning_tokens", 0)
    #             )

    #             usage = AdalFlowResponseUsage(
    #                 input_tokens=completion.usage.input_tokens,
    #                 input_tokens_details=input_tokens_details,
    #                 output_tokens=completion.usage.output_tokens,
    #                 output_tokens_details=output_tokens_details,
    #                 total_tokens=completion.usage.total_tokens,
    #             )
    #         # TODO implement __aiter__ and __iter__ for Response iterator
    #         elif hasattr(completion, '__aiter__') or hasattr(completion, '__iter__'):
    #             # Handle async generator or regular generator/iterator of response stream events
    #             # For generators/iterators, we cannot consume them here as it would exhaust them
    #             # The usage information will need to be tracked elsewhere when the stream is consumed
    #             log.warning("Cannot track usage for generator/iterator. Usage tracking should be handled when consuming the stream.")
    #             return CompletionUsage(
    #                 completion_tokens=None, prompt_tokens=None, total_tokens=None
    #             )
    #         else:
    #             # Handle ChatCompletion with CompletionUsage structure
    #             usage = CompletionUsage(
    #                 completion_tokens=completion.usage.completion_tokens,
    #                 prompt_tokens=completion.usage.prompt_tokens,
    #                 total_tokens=completion.usage.total_tokens,
    #             )
    #         return usage
    #     except Exception as e:
    #         log.error(f"Error tracking the completion usage: {e}")
    #         return CompletionUsage(
    #             completion_tokens=None, prompt_tokens=None, total_tokens=None
    #         )

    # NEW RESPONSE API ONLY FUNCTION
    def track_completion_usage(
        self,
        completion: Union[Response, AsyncIterable],
    ) -> ResponseUsage:
        """Track usage for Response API only."""
        if isinstance(completion, Response):
            # Handle Response object with ResponseUsage structure
            input_tokens_details = InputTokensDetails(
                cached_tokens=getattr(completion.usage, "cached_tokens", 0)
            )

            output_tokens_details = OutputTokensDetails(
                reasoning_tokens=getattr(completion.usage, "reasoning_tokens", 0)
            )

            return AdalFlowResponseUsage(
                input_tokens=completion.usage.input_tokens,
                input_tokens_details=input_tokens_details,
                output_tokens=completion.usage.output_tokens,
                output_tokens_details=output_tokens_details,
                total_tokens=completion.usage.total_tokens,
            )

        # otherwise return the AdalFlowResponseUsage with None values with log warnings
        elif hasattr(completion, "__aiter__") or hasattr(completion, "__iter__"):
            log.warning(
                "Cannot track usage for generator/iterator. Usage tracking should be handled when consuming the stream."
            )
        else:
            log.warning(f"Unknown completion type: {type(completion)}")

        return AdalFlowResponseUsage(
            input_tokens=None,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=None,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=None,
        )

    def parse_embedding_response(
        self, response: CreateEmbeddingResponse
    ) -> EmbedderOutput:
        r"""Parse the embedding response to a structure Adalflow components can understand.

        Should be called in ``Embedder``.
        """
        try:
            return parse_embedding_response(response)
        except Exception as e:
            log.error(f"Error parsing the embedding response: {e}")
            return EmbedderOutput(data=[], error=str(e), raw_response=response)

    def _convert_llm_inputs_to_messages(
        self,
        input: Optional[Any] = None,
        images: Optional[Any] = None,
        detail: Optional[str] = "auto",
    ) -> List[Dict[str, str]]:
        # convert input to messages
        messages: List[Dict[str, str]] = []
        if self._input_type == "messages":
            system_start_tag = "<START_OF_SYSTEM_PROMPT>"
            system_end_tag = "<END_OF_SYSTEM_PROMPT>"
            user_start_tag = "<START_OF_USER_PROMPT>"
            user_end_tag = "<END_OF_USER_PROMPT>"

            # new regex pattern to ignore special characters such as \n
            pattern = (
                rf"{system_start_tag}\s*(.*?)\s*{system_end_tag}\s*"
                rf"{user_start_tag}\s*(.*?)\s*{user_end_tag}"
            )

            # Compile the regular expression
            regex = re.compile(pattern, re.DOTALL)

            # re.DOTALL is to allow . to match newline so that (.*?) does not match in a single line
            regex = re.compile(pattern, re.DOTALL)
            # Match the pattern
            match = regex.match(input)
            system_prompt, input_str = None, None

            if match:
                system_prompt = match.group(1)
                input_str = match.group(2)
            else:
                print("No match found.")
            if system_prompt and input_str:
                messages.append({"role": "system", "content": system_prompt})
                if images:
                    content = [{"type": "text", "text": input_str}]
                    if isinstance(images, (str, dict)):
                        images = [images]
                    for img in images:
                        content.append(self._prepare_image_content(img, detail))
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": "user", "content": input_str})
        if len(messages) == 0:
            if images:
                content = [{"type": "text", "text": input}]
                if isinstance(images, (str, dict)):
                    images = [images]
                for img in images:
                    content.append(self._prepare_image_content(img, detail))
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "system", "content": input})
        return messages

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        r"""
        Specify the API input type and output api_kwargs that will be used in _call and _acall methods.
        Convert the Component's standard input, and system_input(chat model) and model_kwargs into API-specific format.
        For multimodal inputs, images can be provided in model_kwargs["images"] as a string path, URL, or list of them.
        The model specified in model_kwargs["model"] must support multimodal capabilities when using images.

        Args:
            input: The input text or messages to process
            model_kwargs: Additional parameters including:
                - images: Optional image source(s) as path, URL, or list of them
                - detail: Image detail level ('auto', 'low', or 'high'), defaults to 'auto'
                - model: The model to use (must support multimodal inputs if images are provided)
            model_type: The type of model (EMBEDDER or LLM)

        Returns:
            Dict: API-specific kwargs for the model call
        """

        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            # convert input to input
            if not isinstance(input, Sequence):
                raise TypeError("input must be a sequence of text")
            final_model_kwargs["input"] = input
        # elif model_type == ModelType.LLM or model_type == ModelType.LLM_REASONING:
        #     # convert input to messages
        #     messages: List[Dict[str, str]] = []
        #     images = final_model_kwargs.pop("images", None)
        #     detail = final_model_kwargs.pop("detail", "auto")
        #     messages = self._convert_llm_inputs_to_messages(input, images, detail)
        #     final_model_kwargs["messages"] = messages

        # replacing to use the model response API and save under input
        elif model_type == ModelType.LLM or model_type == ModelType.LLM_REASONING:
            # For Response API, input should be a string, not messages
            # If input is already a string, use it directly
            # if isinstance(input, str):
            #     final_model_kwargs["input"] = input
            # else:
            #     # Convert structured input to string format if needed
            #     final_model_kwargs["input"] = self._convert_llm_inputs_to_messages(input)
            final_model_kwargs["input"] = input
            # double check that this is cast to string
        elif model_type == ModelType.IMAGE_GENERATION:
            # For image generation, input is the prompt
            final_model_kwargs["prompt"] = input
            # Ensure model is specified
            if "model" not in final_model_kwargs:
                raise ValueError("model must be specified for image generation")
            # Set defaults for DALL-E 3 if not specified
            final_model_kwargs["size"] = final_model_kwargs.get("size", "1024x1024")
            final_model_kwargs["quality"] = final_model_kwargs.get(
                "quality", "standard"
            )
            final_model_kwargs["n"] = final_model_kwargs.get("n", 1)
            final_model_kwargs["response_format"] = final_model_kwargs.get(
                "response_format", "url"
            )

            # Handle image edits and variations
            image = final_model_kwargs.get("image")
            if isinstance(image, str) and os.path.isfile(image):
                final_model_kwargs["image"] = self._encode_image(image)

            mask = final_model_kwargs.get("mask")
            if isinstance(mask, str) and os.path.isfile(mask):
                final_model_kwargs["mask"] = self._encode_image(mask)
        else:
            raise ValueError(f"model_type {model_type} is not supported")
        return final_model_kwargs

    def parse_image_generation_response(self, response: List[Image]) -> GeneratorOutput:
        """Parse the image generation response into a GeneratorOutput."""
        try:
            # Extract URLs or base64 data from the response
            data = [img.url or img.b64_json for img in response]
            # For single image responses, unwrap from list
            if len(data) == 1:
                data = data[0]
            return GeneratorOutput(
                data=data,
                raw_response=str(response),
            )
        except Exception as e:
            log.error(f"Error parsing image generation response: {e}")
            return GeneratorOutput(data=None, error=str(e), raw_response=str(response))

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
        kwargs is the combined input and model_kwargs.  Support streaming call.
        For reasoning model, users can add "reasoning" key to the api_kwargs to pass the reasoning config.
        eg:
        model_kwargs = {
            "model": "gpt-4o-reasoning",
            "reasoning": {
                "effort": "medium", # low, medium, highc
                "summary": "auto", #detailed, auto, none
            }
        }
        """
        self._api_kwargs = api_kwargs
        if model_type == ModelType.EMBEDDER:
            return self.sync_client.embeddings.create(**api_kwargs)
        # OLD CHAT COMPLETION CALLS (COMMENTED OUT)
        # elif model_type == ModelType.LLM:
        #     if "stream" in api_kwargs and api_kwargs.get("stream", False):
        #         log.debug("streaming call")
        #         self.chat_completion_parser = self.streaming_chat_completion_parser
        #         return self.sync_client.chat.completions.create(**api_kwargs)
        #     else:
        #         log.debug("non-streaming call")
        #         self.chat_completion_parser = self.non_streaming_chat_completion_parser
        #         return self.sync_client.chat.completions.create(**api_kwargs)
        elif model_type == ModelType.LLM_REASONING or model_type == ModelType.LLM:
            if "stream" in api_kwargs and api_kwargs.get("stream", False):
                log.debug("streaming call")
                self.response_parser = (
                    self.streaming_response_parser_sync
                )  # Use sync streaming parser
                return self.sync_client.responses.create(**api_kwargs)
            else:
                log.debug("non-streaming call")
                self.response_parser = self.non_streaming_response_parser
                return self.sync_client.responses.create(**api_kwargs)
        elif model_type == ModelType.IMAGE_GENERATION:
            # Determine which image API to call based on the presence of image/mask
            if "image" in api_kwargs:
                if "mask" in api_kwargs:
                    # Image edit
                    response = self.sync_client.images.edit(**api_kwargs)
                else:
                    # Image variation
                    response = self.sync_client.images.create_variation(**api_kwargs)
            else:
                # Image generation
                response = self.sync_client.images.generate(**api_kwargs)
            return response.data
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
        kwargs is the combined input and model_kwargs. Support async streaming call.

        This method now relies on the OpenAI Responses API to handle streaming and non-streaming calls
        with the asynchronous client
        """
        # store the api kwargs in the client
        self._api_kwargs = api_kwargs
        if self.async_client is None:
            self.async_client = self.init_async_client()
        if model_type == ModelType.EMBEDDER:
            return await self.async_client.embeddings.create(**api_kwargs)
        # old chat completions api calls (commented out)
        # elif model_type == ModelType.LLM:
        #     return await self.async_client.chat.completions.create(**api_kwargs)
        # elif model_type == ModelType.LLM_REASONING:
        #     if "stream" in api_kwargs and api_kwargs.get("stream", False):
        #         log.debug("async streaming call")
        #         self.response_parser = self.streaming_response_parser
        #         # setting response parser as async streaming parser for Response API
        #         return await self.async_client.responses.create(**api_kwargs)
        #     else:
        #         log.debug("async non-streaming call")
        #         self.response_parser = self.non_streaming_response_parser
        #         # setting response parser as async non-streaming parser for Response API
        #         return await self.async_client.responses.create(**api_kwargs)
        elif model_type == ModelType.LLM or model_type == ModelType.LLM_REASONING:
            if "stream" in api_kwargs and api_kwargs.get("stream", False):
                log.debug("async streaming call")
                self.response_parser = (
                    self.streaming_response_parser_async
                )  # Use async streaming parser
                # setting response parser as async streaming parser for Response API
                return await self.async_client.responses.create(**api_kwargs)
            else:
                log.debug("async non-streaming call")
                self.response_parser = self.non_streaming_response_parser
                # setting response parser as async non-streaming parser for Response API
                return await self.async_client.responses.create(**api_kwargs)
        elif model_type == ModelType.IMAGE_GENERATION:
            # Determine which image API to call based on the presence of image/mask
            if "image" in api_kwargs:
                if "mask" in api_kwargs:
                    # Image edit
                    response = await self.async_client.images.edit(**api_kwargs)
                else:
                    # Image variation
                    response = await self.async_client.images.create_variation(
                        **api_kwargs
                    )
            else:
                # Image generation
                response = await self.async_client.images.generate(**api_kwargs)
            return response.data
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

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string.

        Args:
            image_path: Path to image file.

        Returns:
            Base64 encoded image string.

        Raises:
            ValueError: If the file cannot be read or doesn't exist.
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            raise ValueError(f"Image file not found: {image_path}")
        except PermissionError:
            raise ValueError(f"Permission denied when reading image file: {image_path}")
        except Exception as e:
            raise ValueError(f"Error encoding image {image_path}: {str(e)}")

    def _prepare_image_content(
        self, image_source: Union[str, Dict[str, Any]], detail: str = "auto"
    ) -> Dict[str, Any]:
        """Prepare image content for API request.

        Args:
            image_source: Either a path to local image or a URL.
            detail: Image detail level ('auto', 'low', or 'high').

        Returns:
            Formatted image content for API request.
        """
        if isinstance(image_source, str):
            if image_source.startswith(("http://", "https://")):
                return {
                    "type": "image_url",
                    "image_url": {"url": image_source, "detail": detail},
                }
            else:
                base64_image = self._encode_image(image_source)
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": detail,
                    },
                }
        return image_source


# Example usage:
if __name__ == "__main__":
    from adalflow.core import Generator
    from adalflow.utils import setup_env

    # log = get_logger(level="DEBUG")

    setup_env()
    prompt_kwargs = {"input_str": "What is the meaning of life?"}

    gen = Generator(
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-3.5-turbo", "stream": False},
    )
    gen_response = gen(prompt_kwargs)
    print(f"gen_response: {gen_response}")

    # for genout in gen_response.data:
    #     print(f"genout: {genout}")

    # test that to_dict and from_dict works
    # model_client = OpenAIClient()
    # model_client_dict = model_client.to_dict()
    # from_dict_model_client = OpenAIClient.from_dict(model_client_dict)
    # assert model_client_dict == from_dict_model_client.to_dict()


if __name__ == "__main__":

    def test_openai_llm():
        import adalflow as adal

        # setup env or pass the api_key
        from adalflow.utils import setup_env

        setup_env()

        openai_llm = adal.Generator(
            model_client=adal.OpenAIClient(), model_kwargs={"model": "gpt-3.5-turbo"}
        )
        resopnse = openai_llm(prompt_kwargs={"input_str": "What is LLM?"})
        print(resopnse)

    def test_openai_reasoning():
        import adalflow as adal

        # setup env or pass the api_key
        from adalflow.utils import setup_env

        setup_env()

        from adalflow.core.types import ModelType

        openai_llm = adal.Generator(
            model_client=adal.OpenAIClient(),
            model_type=ModelType.LLM_REASONING,
            model_kwargs={
                "model": "o3",
                "reasoning": {"effort": "medium", "summary": "auto"},
            },
        )

        resopnse = openai_llm(prompt_kwargs={"input_str": "What is LLM?"})
        print(resopnse)

    test_openai_reasoning()
