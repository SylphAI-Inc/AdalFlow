import os
from typing import Dict, Sequence, Optional, Any
import backoff

try:
    import groq
except ImportError:
    raise ImportError("Please install groq with: pip install groq")

from groq import Groq, AsyncGroq
from groq import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
)

from core.api_client import APIClient, API_INPUT_TYPE
from core.data_classes import ModelType


class GroqAPIClient(APIClient):
    __doc__ = r"""A component wrapper for the Groq API client.

    Visit https://console.groq.com/docs/ for more api details.
    """

    def __init__(self, api_key: Optional[str] = None):
        r"""It is recommended to set the GROQ_API_KEY environment variable instead of passing it as an argument.

        Args:
            api_key (Optional[str], optional): Groq API key. Defaults to None.
        """
        super().__init__()
        self._api_key = api_key
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
            # "llama2-70b-4096": {
            #     "developer": "Meta",
            #     "context_size": "4096",
            # },
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
        api_key = self._api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Environment variable GROQ_API_KEY must be set")
        self.sync_client = Groq(api_key=api_key)

    def _init_async_client(self):
        api_key = self._api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Environment variable GROQ_API_KEY must be set")
        self.async_client = AsyncGroq(api_key=api_key)

    def parse_chat_completion(self, completion: Any) -> str:
        """
        Parse the completion to a structure your sytem standarizes. (here is str)
        # TODO: standardize the completion
        """
        return completion.choices[0].message.content

    def convert_inputs_to_api_kwargs(
        self,
        input: API_INPUT_TYPE = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.LLM:
            messages: Sequence[Dict[str, str]] = []
            if input is not None and input != "":
                messages.append({"role": "system", "content": input})
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
        ),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        assert (
            "model" in api_kwargs
        ), f"model must be specified in api_kwargs: {api_kwargs}"
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
    async def acall(
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
