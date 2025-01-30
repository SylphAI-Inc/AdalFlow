import asyncio
import time
from adalflow.components.model_client.openai_client import (
    OpenAIClient,
)
from adalflow.core.types import ModelType


from adalflow.utils import setup_env


def benchmark_sync_call(api_kwargs, runs=10):
    """
    Benchmark the synchronous .call() method by running it multiple times.

    Parameters:
    - api_kwargs: The arguments to be passed to the API call
    - runs: The number of times to run the call (default is 10)
    """
    responses = []

    start_time = time.time()

    responses = [
        openai_client.call(
            api_kwargs=api_kwargs,
            model_type=ModelType.LLM,
        )
        for _ in range(runs)
    ]

    end_time = time.time()

    for i, response in enumerate(responses):
        print(f"sync call {i + 1} completed: {response}")

    print(f"\nSynchronous benchmark completed in {end_time - start_time:.2f} seconds")


async def benchmark_async_acall(api_kwargs, runs=10):
    """
    Benchmark the asynchronous .acall() method by running it multiple times concurrently.

    Parameters:
    - api_kwargs: The arguments to be passed to the API call
    - runs: The number of times to run the asynchronous call (default is 10)
    """
    start_time = time.time()

    tasks = [
        openai_client.acall(
            api_kwargs=api_kwargs,
            model_type=ModelType.LLM,
        )
        for _ in range(runs)
    ]

    responses = await asyncio.gather(*tasks)

    end_time = time.time()

    for i, response in enumerate(responses):
        print(f"Async call {i + 1} completed: {response}")

    print(f"\nAsynchronous benchmark completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    setup_env()
    openai_client = OpenAIClient()

    prompt = "Tell me a joke."

    model_kwargs = {"model": "gpt-3.5-turbo", "temperature": 0.5, "max_tokens": 100}
    api_kwargs = openai_client.convert_inputs_to_api_kwargs(
        input=prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM
    )

    print("Starting synchronous benchmark...\n")
    benchmark_sync_call(api_kwargs)

    print("\nStarting asynchronous benchmark...\n")
    asyncio.run(benchmark_async_acall(api_kwargs))
