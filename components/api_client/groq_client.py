from groq import Groq, AsyncGroq, AsyncStream
from groq import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
)
import os
from typing import Dict, Sequence
import backoff

from core.api_client import APIClient
from core.data_classes import ModelType


class GroqAPIClient(APIClient):
    def __init__(self):
        super().__init__()
        self.provider = "Groq"
        self._init_sync_client()
        # https://console.groq.com/docs/models, 4/22/2024
        self.model_lists = {
            "llama3-8b-8192": {
                "developer": "Meta",
                "context_size": "8192",
            },
            "llama3-70b-8192": {
                "developer": "Meta",
                "context_size": "8192",
            },
            "llama2-70b-4096": {
                "developer": "Meta",
                "context_size": "4096",
            },
            "mixtral-8x7b-32768": {
                "developer": "Mistral",
                "context_size": "32768",
            },
            "gemma-7b-it": {
                "developer": "Google",
                "context_size": "8192",
            },
        }
        self.async_client = None  # only initialize if the async call is called

    def _init_sync_client(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Environment variable GROQ_API_KEY must be set")
        self.sync_client = Groq()

    def _init_async_client(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Environment variable GROQ_API_KEY must be set")
        self.async_client = AsyncGroq()

    def _combine_input_and_model_kwargs(
        self,
        input: Sequence,
        combined_model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        final_model_kwargs = combined_model_kwargs.copy()
        if model_type == ModelType.LLM:
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
        ),
        max_time=5,
    )
    def _call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        assert "model" in api_kwargs, "model must be specified"
        assert (
            api_kwargs["model"] in self.model_lists
        ), f"model {api_kwargs['model']} not in the list of available models: {self.model_lists}"
        if model_type == ModelType.LLM:
            completion = self.sync_client.chat.completions.create(**api_kwargs)
            return completion
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
        ),
        max_time=5,
    )
    async def _acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        if self.async_client is None:
            self._init_async_client()
        assert "model" in api_kwargs, "model must be specified"
        assert (
            api_kwargs["model"] in self.model_lists
        ), f"model {api_kwargs['model']} not in the list of available models: {self.model_lists}"
        if model_type == ModelType.LLM:
            completion = await self.async_client.chat.completions.create(**api_kwargs)
            return completion
        else:
            raise ValueError(f"model_type {model_type} is not supported")
