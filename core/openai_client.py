"""
This demonstrates how you can easily use Model to use any api or local model to generate text.
"""

import os
from core.api_client import APIClient
from typing import Any, Dict, Sequence
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
        input: Any,
        combined_model_kwargs: dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> dict:
        r"""
        Convert the Component's standard input and model_kwargs into API-specific format
        """
        final_model_kwargs = combined_model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
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


# import os
# from openai import OpenAI, AsyncOpenAI
# from openai import (
#     APITimeoutError,
#     InternalServerError,
#     RateLimitError,
#     UnprocessableEntityError,
#     BadRequestError,
# )
# import backoff
# from typing import List, Dict, Optional, Any
# from core.prompt_builder import Prompt
# from core.component import Sequential
# from core.generator import Generator, DEFAULT_LIGHTRAG_PROMPT
# from core.api_client import OpenAIClient


# TODO: might combine the OpenAI class wrapper with Generator class into one OpenAIGenerator class
# to share the _init_sync_client method and the backoff decorator
# class OpenAIGenerator(Generator):
#     """
#     All models are stateless and should be used as such.
#     """

#     def __init__(
#         self,
#         prompt: Prompt = Prompt(DEFAULT_LIGHTRAG_PROMPT),
#         model_client: OpenAIClient = OpenAIClient(),
#         output_processors: Optional[Sequential] = None,
#         preset_prompt_kwargs: Optional[Dict] = None,
#         model_kwargs: Optional[Dict] = {},
#     ) -> None:

#         super().__init__(
#             provider="openai",
#             prompt=prompt,
#             output_processors=output_processors,
#             preset_prompt_kwargs=preset_prompt_kwargs,
#             model_kwargs=model_kwargs,
#             model_client=model_client,
#         )

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
#     def call(
#         self,
#         *,
#         input: str,  # process one query
#         prompt_kwargs: Optional[Dict] = {},
#         model_kwargs: Optional[Dict] = {},
#     ) -> str:
#         """
#         input are messages in the format of [{"role": "user", "content": "Hello"}]
#         """

#         composed_model_kwargs = self.compose_model_kwargs(**model_kwargs)
#         if not self.model_client.sync_client:
#             self.model_client.sync_client = OpenAI()

#         prompt_kwargs = self.compose_prompt_kwargs(**prompt_kwargs)
#         # add the input to the prompt kwargs
#         prompt_kwargs["query_str"] = input
#         composed_messages = self.compose_model_input(**prompt_kwargs)
#         print(f"composed_messages: {composed_messages}")
#         print(f"composed_model_kwargs: {composed_model_kwargs}")
#         # completion = self.sync_client.chat.completions.create(
#         #     messages=composed_messages, **composed_model_kwargs
#         # )
#         completion = self.model_client.call(
#             input=composed_messages, **composed_model_kwargs
#         )
#         response = self.parse_completion(completion)
#         if self.output_processors:
#             response = self.output_processors.call(response)
#         return response

#     @backoff.on_exception(
#         backoff.expo,
#         (
#             APITimeoutError,
#             InternalServerError,
#             RateLimitError,
#             UnprocessableEntityError,
#         ),
#         max_time=5,
#     )
#     async def acall(
#         self, messages: List[Dict], model: Optional[str] = None, **kwargs
#     ) -> str:
#         # TODO: add support for acall

#         combined_kwargs = self.compose_model_kwargs(**kwargs)
#         if not self.async_client:
#             self.async_client = AsyncOpenAI()
#         completion = await self.async_client.chat.completions.create(
#             messages=messages, **combined_kwargs
#         )
#         response = self.parse_completion(completion)
#         return response
