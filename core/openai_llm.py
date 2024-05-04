"""
This demonstrates how you can easily use Model to use any api or local model to generate text.
"""

from core.model import Model, ModelType
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
from typing import List, Dict, Optional
from core.prompt_builder import Prompt

DEFAULT_LIGHTRAG_PROMPT = r"""
<<SYS>>
{{task_desc_str}}

{{tools_str}}

{{example_str}}
<</SYS>>
---------------------
User query: {{query_str}}

{% if context_str %}
Context: {{context_str}}
{% endif %}

{% if chat_history_str %}
{{chat_history_str}}
{% endif %}

{% if steps_str %}
{{steps_str}}
{% endif %}

You:
"""


class OpenAIGenerator(Model):
    """
    All models are stateless and should be used as such.
    """

    def __init__(
        self,
        provider: str = "openai",
        prompt: Optional[Prompt] = Prompt(DEFAULT_LIGHTRAG_PROMPT),
        **model_kwargs
    ) -> None:
        type = ModelType.LLM
        super().__init__(provider, type, **model_kwargs)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Environment variable OPENAI_API_KEY must be set")
        self.sync_client = OpenAI()
        self.async_client = None  # only initialize when needed
        self.prompt = prompt

    def compose_model_input(self, **kwargs) -> List[Dict]:

        return super()._componse_lm_input_chat(**kwargs)

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
        input: str,  # process one query
        context_str: Optional[str] = None,
        # query_str: Optional[str] = None,
        task_desc_str: Optional[str] = None,
        chat_history_str: Optional[str] = None,
        tools_str: Optional[str] = None,
        example_str: Optional[str] = None,
        steps_str: Optional[str] = None,
        **model_kwargs
    ) -> str:
        """
        input are messages in the format of [{"role": "user", "content": "Hello"}]
        """

        composed_model_kwargs = self.compose_model_kwargs(**model_kwargs)
        if not self.sync_client:
            self.sync_client = OpenAI()
        prompt_kwargs = {
            "context_str": context_str,
            "query_str": input,
            "task_desc_str": task_desc_str,
            "chat_history_str": chat_history_str,
            "tools_str": tools_str,
            "example_str": example_str,
            "steps_str": steps_str,
        }
        composed_messages = self.compose_model_input(**prompt_kwargs)
        completion = self.sync_client.chat.completions.create(
            messages=composed_messages, **composed_model_kwargs
        )
        # print(f"completion: {completion}")
        response = self.parse_completion(completion)
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
        if model:
            self.model = model

        combined_kwargs = self.combine_model_kwargs(**kwargs)
        if not self.async_client:
            self.async_client = AsyncOpenAI()
        completion = await self.async_client.chat.completions.create(
            messages=messages, **combined_kwargs
        )
        response = self.parse_completion(completion)
        return response
