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
        print(f"model_kwargs: {model_kwargs}    ")

        super().__init__(
            provider="openai",
            prompt=prompt,
            output_processors=output_processors,
            preset_prompt_kwargs=preset_prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        self._init_sync_client()
        # check
        print(f"OpenAIGenerator initialized ")

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
        *,  # start of input and prompt kwargs
        input: str,  # process one query
        # context_str: Optional[str] = None,
        # # query_str: Optional[str] = None,
        # task_desc_str: Optional[str] = None,
        # chat_history_str: Optional[str] = None,
        # tools_str: Optional[str] = None,
        # example_str: Optional[str] = None,
        # steps_str: Optional[str] = None,
        # **prompt_kwargs,  # end of input and prompt kwargs
        prompt_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
    ) -> str:
        """
        input are messages in the format of [{"role": "user", "content": "Hello"}]
        """

        composed_model_kwargs = self.compose_model_kwargs(**model_kwargs)
        if not self.sync_client:
            self.sync_client = OpenAI()
        # prompt_kwargs = {
        #     "context_str": context_str,
        #     "query_str": input,
        #     "task_desc_str": task_desc_str,
        #     "chat_history_str": chat_history_str,
        #     "tools_str": tools_str,
        #     "example_str": example_str,
        #     "steps_str": steps_str,
        # }
        # compose the prompt kwargs
        prompt_kwargs = self.compose_prompt_kwargs(**prompt_kwargs)
        # add the input to the prompt kwargs
        prompt_kwargs["query_str"] = input
        composed_messages = self.compose_model_input(**prompt_kwargs)
        completion = self.sync_client.chat.completions.create(
            messages=composed_messages, **composed_model_kwargs
        )
        # print(f"completion: {completion}")
        response = self.parse_completion(completion)
        # apply output processors
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
