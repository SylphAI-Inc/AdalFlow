"""OpenAI ModelClient integration."""

import os
from typing import Dict, Sequence, Optional, List, Any, TypeVar
import logging
from lightrag.utils.registry import EntityMapping

try:
    import openai
    from openai import OpenAI, AsyncOpenAI
    from openai import (
        APITimeoutError,
        InternalServerError,
        RateLimitError,
        UnprocessableEntityError,
        BadRequestError,
    )

    from openai.types import Completion, CreateEmbeddingResponse
except ImportError:
    raise ImportError("Please install openai with: pip install openai")


from lightrag.core.model_client import ModelClient, API_INPUT_TYPE
from lightrag.core.types import ModelType, EmbedderOutput
from lightrag.core.data_components import parse_embedding_response

import backoff

log = logging.getLogger(__name__)
T = TypeVar("T")


class OpenAIClient(ModelClient):
    __doc__ = r"""A component wrapper for the OpenAI API client.

    Support both embedding and chat completion API. 

    Users (1) simplify use ``Embedder`` and ``Generator`` components by passing OpenAIClient() as the model_client.
    (2) can use this as an example to create their own API client or extend this class(copying and modifing the code) in their own project.

    Note:
        We suggest users not to use `response_format` to enforce output data type or `tools` and `tool_choice`  in your model_kwargs when calling the API. 
        We do not know how OpenAI is doing the formating or what prompt they have added.
        Instead
        - use :ref:`OutputParser<components-output_parsers>` for response parsing and formating.

    References:
        - Embeddings models: https://platform.openai.com/docs/guides/embeddings
        - Chat models: https://platform.openai.com/docs/guides/text-generation
        - OpenAI docs: https://platform.openai.com/docs/introduction
    """

    def __init__(self, api_key: Optional[str] = None):
        r"""It is recommended to set the OPENAI_API_KEY environment variable instead of passing it as an argument.

        Args:
            api_key (Optional[str], optional): OpenAI API key. Defaults to None.
        """
        super().__init__()
        self._api_key = api_key
        self.sync_client = self.init_sync_client()
        self.async_client = None  # only initialize if the async call is called

    def init_sync_client(self):
        api_key = self._api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Environment variable OPENAI_API_KEY must be set")
        return OpenAI(api_key=api_key)

    def init_async_client(self):
        api_key = self._api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Environment variable OPENAI_API_KEY must be set")
        return AsyncOpenAI(api_key=api_key)

    def parse_chat_completion(self, completion: Completion) -> str:
        """Parse the completion to a str."""
        return completion.choices[0].message.content

    def parse_embedding_response(
        self, response: CreateEmbeddingResponse
    ) -> EmbedderOutput:
        r"""Parse the embedding response to a structure LightRAG components can understand.

        Should be called in ``Embedder``.
        """
        try:
            return parse_embedding_response(response)
        except Exception as e:
            log.error(f"Error parsing the embedding response: {e}")
            return EmbedderOutput(data=[], error=str(e), raw_response=response)

    def convert_inputs_to_api_kwargs(
        self,
        input: API_INPUT_TYPE = None,  # user input
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        r"""
        Specify the API input type and output api_kwargs that will be used in _call and _acall methods.
        Convert the Component's standard input, and system_input(chat model) and model_kwargs into API-specific format
        """
        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            # convert input to input
            assert isinstance(input, Sequence), "input must be a sequence of text"
            final_model_kwargs["input"] = input
        elif model_type == ModelType.LLM:
            # convert input to messages
            messages: List[Dict[str, str]] = []
            if input is not None and input != "":
                messages.append({"role": "system", "content": input})
            assert isinstance(
                messages, Sequence
            ), "input must be a sequence of messages"
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
            BadRequestError,
        ),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """
        kwargs is the combined input and model_kwargs
        """
        log.info(f"api_kwargs: {api_kwargs}")
        if model_type == ModelType.EMBEDDER:
            return self.sync_client.embeddings.create(**api_kwargs)
        elif model_type == ModelType.LLM:
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
        """
        kwargs is the combined input and model_kwargs
        """
        if self.async_client is None:
            self.async_client = self.init_async_client()
        if model_type == ModelType.EMBEDDER:
            return await self.async_client.embeddings.create(**api_kwargs)
        elif model_type == ModelType.LLM:
            return await self.async_client.chat.completions.create(**api_kwargs)
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
        exclude = ["sync_client", "async_client"]  # unserializable object
        output = super().to_dict(exclude=exclude)
        return output
