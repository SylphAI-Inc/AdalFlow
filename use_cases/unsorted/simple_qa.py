import asyncio
import time

from adalflow.core.generator import Generator
from adalflow.core.component import Component
from adalflow.components.model_client import OpenAIClient
from adalflow.components.model_client import GroqAPIClient
from adalflow.components.model_client import AnthropicAPIClient


class SimpleQA(Component):
    r"""
    User-defined component who wants to switch between providers like OpenAI and Groq.
    """

    def __init__(
        self, provider: str = "openai", model_kwargs: dict = {"model": "gpt-3.5-turbo"}
    ):
        super().__init__()
        if provider == "openai":
            model_client = OpenAIClient()
        elif provider == "groq":
            model_client = GroqAPIClient()
        elif provider == "anthropic":
            model_client = AnthropicAPIClient()
        else:
            raise ValueError(f"Unknown provider: {provider}")
        self.generator = Generator(model_client=model_client, model_kwargs=model_kwargs)
        self.generator.print_prompt()

    def call(self, query: str) -> str:
        return self.generator({"input_str": query})

    async def acall(self, query: str) -> str:
        return await self.generator.acall({"input_str": query})


if __name__ == "__main__":
    query = "What is the capital of France?"
    queries = [query] * 10

    providers = ["openai", "groq", "anthropic"]
    model_kwargs_list = [
        {"model": "gpt-3.5-turbo"},
        {"model": "llama3-8b-8192"},
        {"model": "claude-3-opus-20240229", "max_tokens": 1000},
    ]
    for provider, model_kwargs in zip(providers, model_kwargs_list):
        simple_qa = SimpleQA(provider=provider, model_kwargs=model_kwargs)
        print(simple_qa)
        t0 = time.time()
        print(simple_qa.call("What is the capital of France?"))
        t1 = time.time()
        print(f"Total time for 1 sync call: {t1 - t0} seconds")

        async def make_async_call(query: str) -> str:
            return await simple_qa.acall(query)

        async def main():
            queries = ["What is the capital of France?"] * 10
            tasks = [make_async_call(query) for query in queries]

            start_time = time.time()
            _ = await asyncio.gather(*tasks)
            end_time = time.time()

            print(f"Total time for 10 async calls: {end_time - start_time} seconds")

        # Execute the asynchronous function
        asyncio.run(main())
