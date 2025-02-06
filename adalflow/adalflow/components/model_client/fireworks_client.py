from typing import Optional, Callable, Any, Literal
from openai.types import Completion
from adalflow.components.model_client.openai_client import OpenAIClient

BASE_URL = "https://api.fireworks.ai/inference/v1/"

__doc__ = r"""
A component wrapper for Fireworks AI's OpenAI-compatible API.

This class extends :class:`OpenAIClient` by customizing several key parameters:
  - Sets the API base URL to ``"https://api.fireworks.ai/inference/v1/"``.
  - Uses the environment variable ``"FIREWORKS_API_KEY"`` to obtain the API key.
  - Defaults the input type to ``"messages"``, which is suitable for multi-turn chat interactions.

**Example usage with AdalFlow Generator:**

.. code-block:: python

    from dotenv import load_dotenv
    from adalflow.core import Generator
    from adalflow.components.model_client.fireworks_client import FireworksClient

    # Load environment variables from a .env file if present
    load_dotenv()

    # Create a FireworksClient instance, optionally passing an API key.
    fireworks_client = FireworksClient(api_key="your_api_key_here")

    # Initialize the Generator with the FireworksClient and desired model parameters.
    generator = Generator(
        model_client=fireworks_client,
        model_kwargs={
            "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "temperature": 0.7,
        }
    )

    # Define the input prompt for the generator.
    prompt_kwargs = {
        "input_str": "Hello from Fireworks AI! Can you summarize the concept of quantum mechanics?"
    }

    # Generate a response using the configured generator.
    response = generator(prompt_kwargs)

    # Check for errors and print the output.
    if response.error:
        print(f"[Fireworks] Generator error: {response.error}")
    else:
        print("[Fireworks] LLM output:")
        print(response.data)
"""


class FireworksClient(OpenAIClient):
    r"""
    A component wrapper for Fireworks AI's OpenAI-compatible API.

    Inherits from :class:`OpenAIClient` but customizes:
      - ``base_url`` to ``"https://api.fireworks.ai/inference/v1/"``
      - ``env_api_key_name`` to ``"FIREWORKS_API_KEY"``
      - Default ``input_type`` to ``"messages"`` for multi-turn chat usage.

    See the module-level documentation for an example of how to use
    :class:`FireworksClient` with the AdalFlow Generator.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: str = BASE_URL,
        env_api_key_name: str = "FIREWORKS_API_KEY",
    ):
        super().__init__(
            api_key=api_key,
            chat_completion_parser=chat_completion_parser,
            input_type=input_type,
            base_url=base_url,
            env_api_key_name=env_api_key_name,
        )


if __name__ == "__main__":
    from dotenv import load_dotenv
    from adalflow.core import Generator
    from adalflow.utils import get_logger

    # Load environment variables from a .env file
    load_dotenv()
    log = get_logger(level="DEBUG")

    # Instantiate the FireworksClient; API key can be read from the environment.
    fw_client = FireworksClient()

    # Create the generator using the FireworksClient and custom model parameters.
    generator = Generator(
        model_client=fw_client,
        model_kwargs={
            "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "temperature": 0.7,
        },
    )

    # Define the prompt to be sent to the model.
    prompt_kwargs = {
        "input_str": "Hello from Fireworks AI! Can you summarize the concept of quantum mechanics?"
    }

    # Generate a response and print the result.
    response = generator(prompt_kwargs)

    if response.error:
        print(f"[Fireworks] Generator error: {response.error}")
    else:
        print("[Fireworks] LLM output:")
        print(response.data)
