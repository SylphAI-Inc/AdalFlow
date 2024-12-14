import asyncio
import time
from adalflow.components.model_client import (
    OpenAIClient,
)  # Assuming OpenAIClient with .call() and .acall() is available
from adalflow.core.types import ModelType

from getpass import getpass
import os

from adalflow.utils import setup_env

# Load environment variables - Make sure to have OPENAI_API_KEY in .env file and .env is present in current folder
if os.path.isfile(".env"):
    setup_env(".env")

# Prompt user to enter their API keys securely
if "OPENAI_API_KEY" not in os.environ:
    openai_api_key = getpass("Please enter your OpenAI API key: ")
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = openai_api_key
    print("API keys have been set.")


# Synchronous function for benchmarking .call()
def benchmark_sync_call(api_kwargs, runs=10):
    """
    Benchmark the synchronous .call() method by running it multiple times.

    Parameters:
    - api_kwargs: The arguments to be passed to the API call
    - runs: The number of times to run the call (default is 10)
    """
    # List to store responses
    responses = []

    # Record the start time of the benchmark
    start_time = time.time()

    # Perform synchronous API calls for the specified number of runs
    responses = [
        openai_client.call(
            api_kwargs=api_kwargs,  # API arguments
            model_type=ModelType.LLM,  # Model type (e.g., LLM for language models)
        )
        for _ in range(runs)  # Repeat 'runs' times
    ]

    # Record the end time after all calls are completed
    end_time = time.time()

    # Output the results of each synchronous call
    for i, response in enumerate(responses):
        print(f"sync call {i + 1} completed: {response}")

    # Print the total time taken for all synchronous calls
    print(f"\nSynchronous benchmark completed in {end_time - start_time:.2f} seconds")


# Asynchronous function for benchmarking .acall()
async def benchmark_async_acall(api_kwargs, runs=10):
    """
    Benchmark the asynchronous .acall() method by running it multiple times concurrently.

    Parameters:
    - api_kwargs: The arguments to be passed to the API call
    - runs: The number of times to run the asynchronous call (default is 10)
    """
    # Record the start time of the benchmark
    start_time = time.time()

    # Create a list of asynchronous tasks for the specified number of runs
    tasks = [
        openai_client.acall(
            api_kwargs=api_kwargs,  # API arguments
            model_type=ModelType.LLM,  # Model type (e.g., LLM for language models)
        )
        for _ in range(runs)  # Repeat 'runs' times
    ]

    # Execute all tasks concurrently and wait for them to finish
    responses = await asyncio.gather(*tasks)

    # Record the end time after all tasks are completed
    end_time = time.time()

    # Output the results of each asynchronous call
    for i, response in enumerate(responses):
        print(f"Async call {i + 1} completed: {response}")

    # Print the total time taken for all asynchronous calls
    print(f"\nAsynchronous benchmark completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # Initialize the OpenAI client
    openai_client = OpenAIClient()

    # Sample prompt for testing
    prompt = "Tell me a joke."

    model_kwargs = {"model": "gpt-3.5-turbo", "temperature": 0.5, "max_tokens": 100}
    api_kwargs = openai_client.convert_inputs_to_api_kwargs(
        input=prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM
    )
    # Run both benchmarks
    print("Starting synchronous benchmark...\n")
    benchmark_sync_call(api_kwargs)

    print("\nStarting asynchronous benchmark...\n")
    asyncio.run(benchmark_async_acall(api_kwargs))
