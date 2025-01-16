"""Azure OpenAI ModelClient integration."""

import os
from typing import Dict, Optional, Any, Callable, Literal
import backoff
import logging

from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, CompletionUsage, GeneratorOutput

# optional import
from adalflow.utils.lazy_import import safe_import, OptionalPackages

openai = safe_import(OptionalPackages.OPENAI.value[0], OptionalPackages.OPENAI.value[1])

from openai import AzureOpenAI, AsyncAzureOpenAI, Stream
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
)
from openai.types.chat import ChatCompletionChunk, ChatCompletion
from adalflow.components.model_client.utils import parse_embedding_response

log = logging.getLogger(__name__)

def get_first_message_content(completion: ChatCompletion) -> str:
    """When we only need the content of the first message.
    It is the default parser for chat completion."""
    return completion.choices[0].message.content

def parse_stream_response(completion: ChatCompletionChunk) -> str:
    """Parse the response of the stream API."""
    return completion.choices[0].delta.content

def handle_streaming_response(generator: Stream[ChatCompletionChunk]):
    """Handle the streaming response."""
    for completion in generator:
        log.debug(f"Raw chunk completion: {completion}")
        parsed_content = parse_stream_response(completion)
        yield parsed_content

class AzureClient(ModelClient):
    """A component wrapper for the Azure OpenAI API client.

    This client supports both chat completion and embedding APIs through Azure OpenAI.
    It can be used with both sync and async operations.

    Args:
        api_key (Optional[str]): Azure OpenAI API key
        api_version (Optional[str]): API version to use
        azure_endpoint (Optional[str]): Azure OpenAI endpoint URL (e.g., https://<resource-name>.openai.azure.com/)
        base_url (Optional[str]): Alternative base URL format (e.g., https://<model-deployment-name>.<region>.models.ai.azure.com)
        chat_completion_parser (Optional[Callable]): Function to parse chat completions
        input_type (Literal["text", "messages"]): Format for input

    Environment Variables:
        AZURE_OPENAI_API_KEY: API key
        AZURE_OPENAI_ENDPOINT: Endpoint URL (new format)
        AZURE_BASE_URL: Base URL (alternative format)
        AZURE_OPENAI_VERSION: API version

    Example:
        >>> from adalflow.components.model_client import AzureClient
        >>> client = AzureClient()
        >>> generator = Generator(
        ...     model_client=client,
        ...     model_kwargs={
        ...         "model": "gpt-4",
        ...         "temperature": 0.7
        ...     }
        ... )
        >>> response = generator({"input_str": "What is the capital of France?"})
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        base_url: Optional[str] = None,
        chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
    ):
        super().__init__()
        self._api_key = api_key
        self._api_version = api_version
        self._azure_endpoint = azure_endpoint
        self._base_url = base_url
        self.sync_client = self.init_sync_client()
        self.async_client = None
        self.chat_completion_parser = chat_completion_parser or get_first_message_content
        self._input_type = input_type

    def _get_endpoint(self) -> str:
        """Get the appropriate endpoint URL based on available configuration."""
        # First try the new format endpoint
        endpoint = self._azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if endpoint:
            return endpoint

        # Then try the alternative base URL format
        base_url = self._base_url or os.getenv("AZURE_BASE_URL")
        if base_url:
            # If base_url is provided in the format https://<model>.<region>.models.ai.azure.com
            # we need to extract the model and region
            if "models.ai.azure.com" in base_url:
                return base_url.rstrip("/")
            # If it's just the model name, construct the full URL
            return f"https://{base_url}.openai.azure.com"

        raise ValueError(
            "Either AZURE_OPENAI_ENDPOINT or AZURE_BASE_URL must be set. "
            "Check your deployment page for a URL like: "
            "https://<resource-name>.openai.azure.com/ or "
            "https://<model-deployment-name>.<region>.models.ai.azure.com"
        )

    def init_sync_client(self):
        api_key = self._api_key or os.getenv("AZURE_OPENAI_API_KEY")
        api_version = self._api_version or os.getenv("AZURE_OPENAI_VERSION")

        if not api_key:
            raise ValueError("Environment variable AZURE_OPENAI_API_KEY must be set")
        if not api_version:
            raise ValueError("Environment variable AZURE_OPENAI_VERSION must be set")

        endpoint = self._get_endpoint()
        
        return AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )

    def init_async_client(self):
        api_key = self._api_key or os.getenv("AZURE_OPENAI_API_KEY")
        api_version = self._api_version or os.getenv("AZURE_OPENAI_VERSION")

        if not api_key:
            raise ValueError("Environment variable AZURE_OPENAI_API_KEY must be set")
        if not api_version:
            raise ValueError("Environment variable AZURE_OPENAI_VERSION must be set")

        endpoint = self._get_endpoint()

        return AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """Convert inputs to Azure OpenAI API kwargs format."""
        final_model_kwargs = model_kwargs.copy()

        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            assert isinstance(input, (list, tuple)), "input must be a sequence of text"
            final_model_kwargs["input"] = input
        elif model_type == ModelType.LLM:
            messages = []
            if input is not None and input != "":
                if self._input_type == "text":
                    messages.append({"role": "system", "content": input})
                else:
                    messages.extend(input)
            final_model_kwargs["messages"] = messages
        else:
            raise ValueError(f"model_type {model_type} is not supported")

        # Ensure model is specified
        if "model" not in final_model_kwargs:
            raise ValueError("model must be specified")

        return final_model_kwargs

    def parse_chat_completion(self, completion: ChatCompletion) -> GeneratorOutput:
        """Parse chat completion response."""
        log.debug(f"completion: {completion}")
        try:
            data = self.chat_completion_parser(completion)
            usage = self.track_completion_usage(completion)
            return GeneratorOutput(data=None, usage=usage, raw_response=data)
        except Exception as e:
            log.error(f"Error parsing completion: {e}")
            return GeneratorOutput(
                data=None, error=str(e), raw_response=str(completion)
            )

    def track_completion_usage(self, completion: ChatCompletion) -> CompletionUsage:
        """Track completion token usage."""
        usage = completion.usage
        return CompletionUsage(
            completion_tokens=usage.completion_tokens,
            prompt_tokens=usage.prompt_tokens,
            total_tokens=usage.total_tokens,
        )

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
        """Make a synchronous call to Azure OpenAI API."""
        log.info(f"api_kwargs: {api_kwargs}")
        if model_type == ModelType.EMBEDDER:
            return self.sync_client.embeddings.create(**api_kwargs)
        elif model_type == ModelType.LLM:
            if "stream" in api_kwargs and api_kwargs.get("stream", False):
                log.debug("streaming call")
                self.chat_completion_parser = handle_streaming_response
                return self.sync_client.chat.completions.create(**api_kwargs)
            return self.sync_client.chat.completions.create(**api_kwargs)
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
        """Make an asynchronous call to Azure OpenAI API."""
        if self.async_client is None:
            self.async_client = self.init_async_client()
        if model_type == ModelType.EMBEDDER:
            return await self.async_client.embeddings.create(**api_kwargs)
        elif model_type == ModelType.LLM:
            return await self.async_client.chat.completions.create(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AzureClient':
        """Create an instance from a dictionary."""
        obj = super().from_dict(data)
        obj.sync_client = obj.init_sync_client()
        obj.async_client = obj.init_async_client()
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """Convert the instance to a dictionary."""
        exclude = ["sync_client", "async_client"]
        output = super().to_dict(exclude=exclude)
        return output 