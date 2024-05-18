# https://docs.anthropic.com/en/api/messages
import os
from typing import Any
import anthropic
from anthropic import (
    RateLimitError,
    APITimeoutError,
    InternalServerError,
    UnprocessableEntityError,
    BadRequestError,
)

from core.api_client import APIClient


class AnthropicAPIClient(APIClient):
    def __init__(self):
        super().__init__()

    def _init_sync_client(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Environment variable ANTHROPIC_API_KEY must be set")
        self.sync_client = anthropic.Anthropic()

    def _init_async_client(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Environment variable ANTHROPIC_API_KEY must be set")
        self.async_client = anthropic.AsyncAnthropic()

    def _combine_input_and_model_kwargs(
        self,
        input: Any,  # user input such as a query or a sequence of str for embeddings
        combined_model_kwargs: dict = {},
    ) -> dict:
        return {
            "message": input,
            **combined_model_kwargs,
        }
