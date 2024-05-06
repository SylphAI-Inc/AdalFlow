from core.generator import Generator
from groq import Groq
from groq import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
)
import os
from typing import Dict, List, Optional
import backoff


class GroqGenerator(Generator):
    name = "GroqGenerator"

    def __init__(self, provider: str, model: str, **kwargs):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Please set the GROQ_API_KEY environment variable")
        super().__init__(provider, model, **kwargs)

        self.provider = provider
        self.model = model
        self.sync_client = Groq()
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
        if model not in self.model_lists:
            raise ValueError(
                f"Model {model} not in the list of available models: {self.model_lists}"
            )
        print(f"Initialized {self.name} with model {self.model}")

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
    def call(self, messages: List[Dict], model: Optional[str] = None, **kwargs) -> str:
        if model:  # overwrite the default model
            self.model = model
        combined_kwargs = self.combine_kwargs(**kwargs)
        if not self.sync_client:
            self.sync_client = Groq()
        completion = self.sync_client.chat.completions.create(
            messages=messages, **combined_kwargs
        )
        response = self.parse_completion(completion)
        return response


if __name__ == "__main__":
    generator = GroqGenerator("groq", "llama3-70b-8192")
    import time

    t0 = time.time()
    response = generator([{"role": "user", "content": "Hello"}])
    print(response)
    print(f"Time: {time.time() - t0}")
