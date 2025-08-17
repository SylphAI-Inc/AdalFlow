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
from dataclasses import dataclass

from adalflow.components.model_client.utils import (
    parse_embedding_response,
    format_content_for_response_api,
)

log = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class ParsedResponseContent:
    """Structured container for parsed response content from OpenAI Response API.

    This dataclass provides a consistent interface for accessing different types
    of content that can be returned by the Response API, including text, images,
    tool calls, reasoning chains, and more.

    Attributes:
        text: The main text content from the response
        images: List of image data (base64 or URLs) from image generation
        tool_calls: List of other tool call results
        reasoning: Reasoning chain from reasoning models
        code_outputs: Outputs from code interpreter
        raw_output: The original output array for advanced processing
    """
    text: Optional[str] = None
    images: Optional[Union[str, List[str]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    reasoning: Optional[List[Dict[str, Any]]] = None
    code_outputs: Optional[List[Dict[str, Any]]] = None
    raw_output: Optional[Any] = None

    def __bool__(self) -> bool:
        """Check if there's any content."""
        return any([
            self.text,
            self.images,
            self.tool_calls,
            self.reasoning,
            self.code_outputs
        ])


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


def parse_response_output(response: Response) -> ParsedResponseContent:
    """Parse response output that may include various types of content and tool calls.

    The output array can contain:
    - Output messages (with nested content items)
    - Tool calls (file search, function, web search, computer use, etc.)
    - Reasoning chains
    - Image generation calls
    - Code interpreter calls
    - And more...

    Returns:
        ParsedResponseContent: Structured content with typed access to all response data
    """
    log.debug(f"raw response from api: {response}")

    content = ParsedResponseContent()

    # Store raw output for advanced users
    if hasattr(response, 'output'):
        content.raw_output = response.output

    # First try to use output_text if available (SDK convenience property)
    if hasattr(response, 'output_text') and response.output_text:
        content.text = response.output_text
    # Parse the output array manually if no output_text
    if hasattr(response, 'output') and response.output:
        parsed = _parse_output_array(response.output)
        content.text = content.text or parsed.get("text")
        content.images = parsed.get("images", [])
        content.tool_calls = parsed.get("tool_calls")
        content.reasoning = parsed.get("reasoning")
        content.code_outputs = parsed.get("code_outputs")

    return content



def _parse_message(item) -> Dict[str, Any]:
    """Parse a message item from the output array.

    Args:
        item: A message item with type="message" and content array

    Returns:
        Dict with parsed text and images from the message
    """
    result = {"text": None}

    if hasattr(item, 'content') and isinstance(item.content, list):
        # now pick the longer response 
        text_parts = []

        for content_item in item.content:
            content_type = getattr(content_item, 'type', None)

            if content_type == "output_text":
                if hasattr(content_item, 'text'):
                    text_parts.append(content_item.text)

        if text_parts:
            result["text"] = max(text_parts, key=len) if len(text_parts) > 1 else text_parts[0]

    return result


def _parse_reasoning(item) -> Dict[str, Any]:
    """Parse a reasoning item from the output array.

    Args:
        item: A reasoning item with type="reasoning" and summary array

    Returns:
        Dict with extracted reasoning text and full structure
    """
    result = {"reasoning": None}

    # Extract text from reasoning summary if available
    if hasattr(item, 'summary') and isinstance(item.summary, list):
        summary_texts = []
        for summary_item in item.summary:
            if hasattr(summary_item, 'type') and summary_item.type == "summary_text":
                if hasattr(summary_item, 'text'):
                    summary_texts.append(summary_item.text)

        if summary_texts:
            # Store reasoning text separately for later combination
            result["reasoning"] = "\n".join(summary_texts)

    return result


def _parse_image(item) -> Dict[str, Any]:
    """Parse an image generation call item from the output array.

    Args:
        item: An image generation item with type="image_generation_call" and result field

    Returns:
        Dict with extracted image data
    """
    result = {"images": None}

    if hasattr(item, 'result'):
        # The result contains the base64 image data or URL
        result["images"] = item.result

    return result


def _parse_tool_call(item) -> Dict[str, Any]:
    """Parse a tool call item from the output array.

    Args:
        item: A tool call item (various types ending in _call or containing tool_call)

    Returns:
        Dict with tool call information
    """
    item_type = getattr(item, 'type', None)

    if item_type == "image_generation_call":
        # Handle image generation - extract the result which contains the image data
        if hasattr(item, 'result'):
            # The result contains the base64 image data or URL
            return {"images": item.result}
    elif item_type == "code_interpreter_tool_call":
        return {"code_outputs": [_serialize_item(item)]}
    else:
        # Generic tool call
        return {
            "tool_calls": [{
                "type": item_type,
                "content": _serialize_item(item)
            }]
        }

    return {}


def _parse_output_array(output_array) -> Dict[str, Any]:
    """Parse the entire output array, processing all elements.

    The output array typically contains:
    1. Reasoning (optional) - thinking/reasoning before the response
    2. Message - the actual response with content
    3. Tool calls (optional) - any tool invocations

    Returns:
        Dict with keys: text, images, tool_calls, reasoning, code_outputs
    """
    result = {
        "text": None,
        "images": None,
        "tool_calls": None,
        "reasoning": None,
        "code_outputs": None
    }

    if not output_array:
        return result

    # Process all items in the array
    all_images = []
    all_tool_calls = []
    all_code_outputs = []
    all_reasoning = None
    text = None

    for item in output_array:
        item_type = getattr(item, 'type', None)

        if item_type == "reasoning":
            # Parse reasoning item
            parsed = _parse_reasoning(item)
            if parsed.get("reasoning"):
                all_reasoning = parsed["reasoning"]

        elif item_type == "message":
            # Parse message item
            parsed = _parse_message(item)
            if parsed.get("text"):
                text = parsed["text"]

        elif item_type == "image_generation_call":
            # Parse image generation call separately
            parsed = _parse_image(item)
            if parsed.get("images"):
                all_images.append(parsed["images"])

        elif item_type and ('call' in item_type or 'tool' in item_type):
            # Parse other tool calls
            parsed = _parse_tool_call(item)
            if parsed.get("tool_calls"):
                all_tool_calls.extend(parsed["tool_calls"])
            if parsed.get("code_outputs"):
                all_code_outputs.extend(parsed["code_outputs"])


    result["text"] = text if text else None # TODO: they can potentially send multiple complete text messages, we might need to save all of them and only return the first that can convert to outpu parser

    # Set other fields if they have content
    result["images"] = all_images
    if all_tool_calls:
        result["tool_calls"] = all_tool_calls
    if all_reasoning:
        result["reasoning"] = all_reasoning
    if all_code_outputs:
        result["code_outputs"] = all_code_outputs

    return result


def _serialize_item(item) -> Dict[str, Any]:
    """Convert an output item to a serializable dict."""
    result = {}
    for attr in dir(item):
        if not attr.startswith('_'):
            value = getattr(item, attr, None)
            if value is not None and not callable(value):
                result[attr] = value
    return result


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




class OpenAIClient(ModelClient):
    __doc__ = r"""A component wrapper for the OpenAI API client.

    Support both embedding and response API, including multimodal capabilities.


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

    Examples:
        Basic text generation::

            from adalflow.components.model_client import OpenAIClient
            from adalflow.core import Generator

            # Initialize client (uses OPENAI_API_KEY env var by default)
            client = OpenAIClient()

            # Create a generator for text
            generator = Generator(
                model_client=client,
                model_kwargs={"model": "gpt-4o-mini"}
            )

            # Generate response
            response = generator(prompt_kwargs={"input_str": "What is machine learning?"})
            print(response.data)

        Multimodal with URL image::

            # Vision model with image from URL
            generator = Generator(
                model_client=OpenAIClient(),
                model_kwargs={
                    "model": "gpt-4o",
                    "images": "https://example.com/chart.jpg"
                }
            )

            response = generator(
                prompt_kwargs={"input_str": "Analyze this chart and explain the trends"}
            )

        Multimodal with local images::

            # Multiple local images
            generator = Generator(
                model_client=OpenAIClient(),
                model_kwargs={
                    "model": "gpt-4o",
                    "images": [
                        "/path/to/image1.jpg",
                        "/path/to/image2.png"
                    ]
                }
            )

            response = generator(
                prompt_kwargs={"input_str": "Compare these two images"}
            )

        Pre-formatted images with custom encoding::

            import base64
            from adalflow.core.functional import encode_image

            # Option 1: Using the encode_image helper
            base64_img = encode_image("/path/to/image.jpg")

            # Option 2: Manual base64 encoding
            with open("/path/to/image.png", "rb") as f:
                base64_img = base64.b64encode(f.read()).decode('utf-8')

            # Use pre-formatted image data
            generator = Generator(
                model_client=OpenAIClient(),
                model_kwargs={
                    "model": "gpt-4o",
                    "images": [
                        # Pre-formatted as base64 data URI
                        f"data:image/png;base64,{base64_img}",
                        # Or as a dict with type and image_url
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_img}"
                        },
                        # Mix with regular URLs
                        "https://example.com/chart.jpg"
                    ]
                }
            )

            response = generator(
                prompt_kwargs={"input_str": "Analyze these images"}
            )

        Reasoning models (O1, O3)::

            from adalflow.core.types import ModelType

            # O3 reasoning model with effort configuration
            generator = Generator(
                model_client=OpenAIClient(),
                model_type=ModelType.LLM_REASONING,
                model_kwargs={
                    "model": "o3",
                    "reasoning": {
                        "effort": "medium",  # low, medium, high
                        "summary": "auto"    # detailed, auto, none
                    }
                }
            )

            response = generator(
                prompt_kwargs={"input_str": "Solve this complex problem: ..."}
            )

        Image generation with DALL-E (legacy method)::

            from adalflow.core.types import ModelType

            # Generate an image using ModelType.IMAGE_GENERATION
            generator = Generator(
                model_client=OpenAIClient(),
                model_type=ModelType.IMAGE_GENERATION,
                model_kwargs={
                    "model": "dall-e-3",
                    "size": "1024x1792",
                    "quality": "hd",
                    "n": 1
                }
            )

            response = generator(
                prompt_kwargs={"input_str": "A futuristic city with flying cars at sunset"}
            )
            # response.data contains the image URL or base64 data

        Image generation via tools (new API)::

            import base64

            # Generate images using the new tools API
            generator = Generator(
                model_client=OpenAIClient(),
                model_kwargs={
                    "model": "gpt-4o-mini",  # or any model that supports tools
                    "tools": [{"type": "image_generation"}]
                }
            )

            # Generate an image
            response = generator(
                prompt_kwargs={
                    "input_str": "Generate an image of a gray tabby cat hugging an otter with an orange scarf"
                }
            )

            # Access the generated image(s)
            if isinstance(response.data, list):
                # Multiple images
                for i, img_base64 in enumerate(response.data):
                    with open(f"generated_{i}.png", "wb") as f:
                        f.write(base64.b64decode(img_base64))
            elif isinstance(response.data, str):
                # Single image
                with open("generated.png", "wb") as f:
                    f.write(base64.b64decode(response.data))
            elif isinstance(response.data, dict) and "images" in response.data:
                # Mixed response with text and images
                print("Text:", response.data["text"])
                for i, img_base64 in enumerate(response.data["images"]):
                    with open(f"generated_{i}.png", "wb") as f:
                        f.write(base64.b64decode(img_base64))

        Embeddings::

            from adalflow.core import Embedder

            # Create embedder
            embedder = Embedder(
                model_client=OpenAIClient(),
                model_kwargs={"model": "text-embedding-3-small"}
            )

            # Generate embeddings
            embeddings = embedder(input=["Hello world", "Machine learning"])
            print(embeddings.data)  # List of embedding vectors

        Streaming responses::

            from adalflow.components.model_client.utils import extract_text_from_response_stream

            # Enable streaming
            generator = Generator(
                model_client=OpenAIClient(),
                model_kwargs={
                    "model": "gpt-4o",
                    "stream": True
                }
            )

            # Stream the response
            response = generator(prompt_kwargs={"input_str": "Tell me a story"})

            # Extract text from Response API streaming events
            for event in response.raw_response:
                text = extract_text_from_response_stream(event)
                if text:
                    print(text, end="")

        Custom API endpoint::

            # Use with third-party providers or local models
            client = OpenAIClient(
                base_url="https://api.custom-provider.com/v1/",
                api_key="your-api-key",
                headers={"X-Custom-Header": "value"}
            )

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
        - OpenAI API Overview: https://platform.openai.com/docs/introduction, https://platform.openai.com/docs/guides/images-vision?api-mode=responses
        - Embeddings Guide: https://platform.openai.com/docs/guides/embeddings
        - Chat Completion Models: https://platform.openai.com/docs/guides/text-generation
        - Response api: https://platform.openai.com/docs/api-reference/responses/create, Analyze images and use them as input and/or generate images as output
        - Vision Models: https://platform.openai.com/docs/guides/vision
        - Image Generation: https://platform.openai.com/docs/guides/images
        - reasoning: https://platform.openai.com/docs/guides/reasoning

    Note:
        - Ensure each OpenAIClient instance is used by one generator only.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        # OLD CHAT COMPLETION PARSER PARAMS (kept for backward compatibility)
        non_streaming_chat_completion_parser: Optional[
            Callable[[Completion], Any]
        ] = None,  # non-streaming parser - deprecated but accepted
        streaming_chat_completion_parser: Optional[
            Callable[[Completion], Any]
        ] = None,  # streaming parser - deprecated but accepted
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
            non_streaming_chat_completion_parser (Optional[Callable[[Completion], Any]], optional): DEPRECATED - Legacy parser for chat completions. Ignored, kept for backward compatibility. Defaults to None.
            streaming_chat_completion_parser (Optional[Callable[[Completion], Any]], optional): DEPRECATED - Legacy parser for streaming chat completions. Ignored, kept for backward compatibility. Defaults to None.
            non_streaming_response_parser (Optional[Callable[[Response], Any]], optional): Parser for non-streaming responses. Defaults to None.
            streaming_response_parser (Optional[Callable[[Response], Any]], optional): Parser for streaming responses. Defaults to None.
            input_type (Literal["text", "messages"]): Input type for the client. Defaults to "text".
            base_url (str): The API base URL to use when initializing the client.
            env_api_key_name (str): The environment variable name for the API key. Defaults to `"OPENAI_API_KEY"`.
            organization (Optional[str], optional): OpenAI organization key. Defaults to None.
            headers (Optional[Dict[str, str]], optional): Additional headers to include in API requests. Defaults to None.
        """
        # Log deprecation warning if old parsers are provided
        if non_streaming_chat_completion_parser is not None:
            log.warning(
                "non_streaming_chat_completion_parser is deprecated and will be ignored. "
                "The OpenAI client now uses the Response API exclusively."
            )
        if streaming_chat_completion_parser is not None:
            log.warning(
                "streaming_chat_completion_parser is deprecated and will be ignored. "
                "The OpenAI client now uses the Response API exclusively."
            )

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

    # NEW RESPONSE API ONLY FUNCTION
    def parse_chat_completion(
        self,
        completion: Union[Response, AsyncIterable],
    ) -> "GeneratorOutput":
        """Parse the Response API completion and put it into the raw_response.
        Fully migrated to Response API only."""

        parser = self.response_parser
        log.info(f"completion/response: {completion}, parser: {parser}")

        # Check if this is a Response with complex output (tools, images, etc.)
        if isinstance(completion, Response):
            parsed_content = parse_response_output(completion)
            usage = self.track_completion_usage(completion)

            data = parsed_content.text

            thinking = None
            if parsed_content.reasoning:
                thinking = str(parsed_content.reasoning)


            return GeneratorOutput(
                data=data,  # only text
                thinking=thinking,
                images=parsed_content.images,  # List of image data (base64 or URLs)
                tool_use=None,  # Will be populated when we handle function tool calls
                error=None,
                raw_response=data,
                usage=usage
            )
        # Regular response handling (streaming or other)
        data = parser(completion)
        usage = self.track_completion_usage(completion)
        return GeneratorOutput(data=None, error=None, raw_response=data, usage=usage)


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
            log.debug(
                "Cannot track usage for generator/iterator. Usage tracking should be handled when consuming the stream."
            )
        else:
            log.debug(f"Unknown completion type: {type(completion)}")

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

    # adapted for the response api
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
        elif model_type == ModelType.LLM or model_type == ModelType.LLM_REASONING:
            # Check if images are provided for multimodal input
            images = final_model_kwargs.pop("images", None)

            if images:
                # Use helper function to format content with images
                content = format_content_for_response_api(input, images)

                # For responses.create API, wrap in user message format
                final_model_kwargs["input"] = [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            else:
                # Text-only input
                final_model_kwargs["input"] = input
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
