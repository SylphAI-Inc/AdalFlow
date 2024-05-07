"""
This demonstrates how you can easily use Model to use any api or local model to generate text.
"""

import os
from core.api_client import APIClient
from typing import Any, Dict, Sequence, Union
from core.data_classes import ModelType
from openai import OpenAI, AsyncOpenAI
import backoff
from openai import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
    BadRequestError,
)


class OpenAIClient(APIClient):
    def __init__(self):
        super().__init__()
        self.provider = "OpenAI"
        self.sync_client = self._init_sync_client()

    def _init_sync_client(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Environment variable OPENAI_API_KEY must be set")
        return OpenAI()

    def _init_async_client(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Environment variable OPENAI_API_KEY must be set")
        return AsyncOpenAI()

    def _combine_input_and_model_kwargs(
        self,
        input: Union[str, Sequence],
        combined_model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        r"""
        Specify the API input type.
        Convert the Component's standard input and model_kwargs into API-specific format
        """
        final_model_kwargs = combined_model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            # convert input to input
            assert isinstance(input, Sequence), "input must be a sequence of text"
            final_model_kwargs["input"] = input
        elif model_type == ModelType.LLM:
            # convert input to messages
            assert isinstance(input, Sequence), "input must be a sequence of messages"
            final_model_kwargs["messages"] = input
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
    def _call(self, kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """
        kwargs is the combined input and model_kwargs
        """
        if model_type == ModelType.EMBEDDER:
            return self.sync_client.embeddings.create(**kwargs)
        elif model_type == ModelType.LLM:
            return self.sync_client.chat.completions.create(**kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")
