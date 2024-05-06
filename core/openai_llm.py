"""
This demonstrates how you can easily use Model to use any api or local model to generate text.
"""

import os
from openai import OpenAI, AsyncOpenAI
from openai import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
    BadRequestError,
)
import backoff
from typing import List, Dict, Optional, Any
from core.prompt_builder import Prompt
from core.component import Sequential
from core.generator import Generator, DEFAULT_LIGHTRAG_PROMPT


# TODO: might combine the OpenAI class wrapper with Generator class into one OpenAIGenerator class
# to share the _init_sync_client method and the backoff decorator
class OpenAIGenerator(Generator):
    """
    All models are stateless and should be used as such.
    """

    def __init__(
        self,
        prompt: Prompt = Prompt(DEFAULT_LIGHTRAG_PROMPT),
        output_processors: Optional[Sequential] = None,
        preset_prompt_kwargs: Optional[Dict] = None,
        model_kwargs: Optional[Dict] = {},
    ) -> None:

        super().__init__(
            provider="openai",
            prompt=prompt,
            output_processors=output_processors,
            preset_prompt_kwargs=preset_prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        self._init_sync_client()

    def _init_sync_client(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Environment variable OPENAI_API_KEY must be set")
        self.sync_client = OpenAI()

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
    def call(
        self,
        *,
        input: str,  # process one query
        prompt_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
    ) -> str:
        """
        input are messages in the format of [{"role": "user", "content": "Hello"}]
        """

        composed_model_kwargs = self.compose_model_kwargs(**model_kwargs)
        if not self.sync_client:
            self.sync_client = OpenAI()

        prompt_kwargs = self.compose_prompt_kwargs(**prompt_kwargs)
        # add the input to the prompt kwargs
        prompt_kwargs["query_str"] = input
        composed_messages = self.compose_model_input(**prompt_kwargs)
        print(f"composed_messages: {composed_messages}")
        print(f"composed_model_kwargs: {composed_model_kwargs}")
        completion = self.sync_client.chat.completions.create(
            messages=composed_messages, **composed_model_kwargs
        )
        response = self.parse_completion(completion)
        if self.output_processors:
            response = self.output_processors.call(response)
        return response

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
        self, messages: List[Dict], model: Optional[str] = None, **kwargs
    ) -> str:
        # TODO: add support for acall

        combined_kwargs = self.compose_model_kwargs(**kwargs)
        if not self.async_client:
            self.async_client = AsyncOpenAI()
        completion = await self.async_client.chat.completions.create(
            messages=messages, **combined_kwargs
        )
        response = self.parse_completion(completion)
        return response
