"""
This demonstrates how to wrap the OpenAI API client to fit into LightRAG APIClient
"""

import os
from core.api_client import APIClient
from typing import Any, Dict, Sequence, Union, Optional, List
from core.data_classes import ModelType
import backoff

try:
    import google.generativeai as genai
    from google.api_core.exceptions import (
        InternalServerError,
        BadRequest,
        GoogleAPICallError,
    )
    from google.generativeai.types import GenerateContentResponse
except ImportError:
    raise ImportError("Please install google-generativeai to use GoogleGenAIClient")


class GoogleGenAIClient(APIClient):
    def __init__(self):
        super().__init__()
        self.sync_client = self._init_sync_client()
        self.async_client = None  # only initialize if the async call is called
        self.tested_llm_models = ["gemini-1.0-pro", "gemini-1.5-pro-latest"]

    def _init_sync_client(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Environment variable GOOGLE_API_KEY must be set")
        genai.configure(api_key=api_key)
        return genai

    # def _init_async_client(self):
    #     api_key = os.getenv("GOOGLE_API_KEY")
    #     if not api_key:
    #         raise ValueError("Environment variable GOOGLE_API_KEY must be set")
    #     return AsyncOpenAI()

    def parse_chat_completion(self, completion: GenerateContentResponse) -> str:
        """
        Parse the completion to a structure your sytem standarizes. (here is str)
        # TODO: standardize the completion
        """
        print(f"completion: {completion}")
        return completion.text
        # result["candidates"][0]["contents"]["parts"][0]["text"]

    def convert_input_to_api_kwargs(
        self,
        input: Union[str, Sequence],
        system_input: Optional[Union[str]] = None,
        combined_model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        r"""
        Specify the API input type and output api_kwargs that will be used in _call and _acall methods.
        Convert the Component's standard input, and system_input(chat model) and model_kwargs into API-specific format
        """
        final_model_kwargs = combined_model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            # convert input to input
            assert isinstance(input, Sequence), "input must be a sequence of text"
            final_model_kwargs["input"] = input
        elif model_type == ModelType.LLM:
            prompt: str = f"{system_input}\n\nUser query: {input}\n You:"

            final_model_kwargs["prompt"] = prompt
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

            config = genai.GenerationConfig(**api_kwargs)
            llm = genai.GenerativeModel(model_name=model, generation_config=config)
            return llm.generate_content(contents=prompt)
        else:
            raise ValueError(f"model_type {model_type} is not supported")
