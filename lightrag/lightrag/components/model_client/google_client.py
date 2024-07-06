"""Google GenAI ModelClient integration."""

import os
from typing import Dict, Sequence, Optional, Any
import backoff


from lightrag.core.model_client import ModelClient
from lightrag.core.types import ModelType

from lightrag.utils.lazy_import import safe_import, OptionalPackages

# optional import
google = safe_import(
    OptionalPackages.GOOGLE_GENERATIVEAI.value[0],
    OptionalPackages.GOOGLE_GENERATIVEAI.value[1],
)
import google.generativeai as genai
from google.api_core.exceptions import (
    InternalServerError,
    BadRequest,
    GoogleAPICallError,
)
from google.generativeai.types import GenerateContentResponse


class GoogleGenAIClient(ModelClient):
    __doc__ = r"""A component wrapper for the Google GenAI API client.

    Visit https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference for more api details.
    """

    def __init__(self, api_key: Optional[str] = None):
        r"""It is recommended to set the GOOGLE_API_KEY environment variable instead of passing it as an argument."""
        super().__init__()
        self._api_key = api_key
        self.sync_client = self.init_sync_client()
        self.async_client = None  # only initialize if the async call is called
        self.tested_llm_models = ["gemini-1.0-pro", "gemini-1.5-pro-latest"]

    def init_sync_client(self):
        api_key = self._api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Environment variable GOOGLE_API_KEY must be set")
        genai.configure(api_key=api_key)
        return genai

    def parse_chat_completion(self, completion: GenerateContentResponse) -> str:
        """
        Parse the completion to a structure your sytem standarizes. (here is str)
        """
        return completion.text

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
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

            final_model_kwargs["prompt"] = input
        else:
            raise ValueError(f"model_type {model_type} is not supported")
        return final_model_kwargs

    @backoff.on_exception(
        backoff.expo,
        (
            InternalServerError,
            BadRequest,
            GoogleAPICallError,
        ),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """
        kwargs is the combined input and model_kwargs
        """

        if model_type == ModelType.LLM:
            # remove model from api_kwargs
            model = api_kwargs.pop("model")
            prompt = api_kwargs.pop("prompt")

            assert model != "", "model must be specified"
            assert prompt != "", "prompt must be specified"

            config = genai.GenerationConfig(**api_kwargs)
            llm = genai.GenerativeModel(model_name=model, generation_config=config)
            return llm.generate_content(contents=prompt)
        else:
            raise ValueError(f"model_type {model_type} is not supported")
