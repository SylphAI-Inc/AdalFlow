from typing import Optional, Any, Callable, Literal
from openai.types import Completion
from adalflow.components.model_client.openai_client import OpenAIClient

BASE_URL = "https://api.sambanova.ai/v1/"

__doc__ = r"""
A component wrapper for SambaNova's OpenAI-compatible API.

This client extends :class:`OpenAIClient` and customizes:
  - ``base_url`` is set to ``"https://api.sambanova.ai/v1/"``.
  - The API key is read from the environment variable ``SAMBANOVA_API_KEY`` if not provided explicitly.
  - The default ``input_type`` is set to ``"messages"`` for multi-turn chat usage.

**Example usage with the AdalFlow Generator:**

.. code-block:: python

    from dotenv import load_dotenv
    from adalflow.core import Generator
    from adalflow.components.model_client.sambanova_client import SambaNovaClient
    from adalflow.utils import get_logger

    # Load environment variables from a .env file.
    load_dotenv()

    # Initialize logging.
    log = get_logger(level="DEBUG")

    # Create a SambaNovaClient instance.
    sn_client = SambaNovaClient()

    # Set up the generator with the SambaNovaClient and model-specific parameters.
    generator = Generator(
        model_client=sn_client,
        model_kwargs={
            "model": "Meta-Llama-3.1-8B-Instruct",
            "temperature": 0.7,
            "top_p": 0.9,
        },
    )

    # Define the input prompt.
    prompt_kwargs = {
        "input_str": "Hello from SambaNova! Can you summarize the concept of quantum computing in simple terms?"
    }

    # Generate the response.
    response = generator(prompt_kwargs)

    if response.error:
        print(f"SambaNova Generator error: {response.error}")
    else:
        print("[SambaNova] LLM output:")
        print(response.data)
"""


class SambaNovaClient(OpenAIClient):
    r"""
    A component wrapper for SambaNova's OpenAI-compatible API.

    This client inherits from :class:`OpenAIClient` and customizes the following:
      - Sets ``base_url`` to ``"https://api.sambanova.ai/v1/"``.
      - Retrieves the API key from the environment variable ``SAMBANOVA_API_KEY`` if not provided.
      - Defaults ``input_type`` to ``"messages"`` to support multi-turn chat interactions.

    The AdalFlow Generator is expected to supply additional model parameters (such as model name,
    temperature, and top_p) via its configuration.

    See the module-level documentation for a complete usage example.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: str = BASE_URL,
        env_api_key_name: str = "SAMBANOVA_API_KEY",
    ):
        """
        Initialize a SambaNovaClient instance.

        :param api_key: (Optional) SambaNova API key. If not provided, the client attempts to read from the
                        environment variable ``SAMBANOVA_API_KEY``.
        :param chat_completion_parser: (Optional) A custom function to parse SambaNova responses.
        :param input_type: Specifies the input format, either ``"text"`` or ``"messages"``. Defaults to ``"messages"``.
        :param base_url: SambaNova API endpoint. Defaults to ``"https://api.sambanova.ai/v1/"``.
        :param env_api_key_name: The name of the environment variable holding the API key. Defaults to ``SAMBANOVA_API_KEY``.
        """
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

    # Load environment variables from a .env file.
    load_dotenv()

    # Set up logging.
    log = get_logger(level="DEBUG")

    # Create an instance of SambaNovaClient.
    sn_client = SambaNovaClient()

    # Configure the generator with the SambaNovaClient and specific model parameters.
    generator = Generator(
        model_client=sn_client,
        model_kwargs={
            "model": "Meta-Llama-3.1-8B-Instruct",
            "temperature": 0.7,
            "top_p": 0.9,
        },
    )

    # Define the prompt to be processed.
    prompt_kwargs = {
        "input_str": "Hello from SambaNova! Can you summarize the concept of quantum computing in simple terms?"
    }

    # Generate the response and display the result.
    response = generator(prompt_kwargs)

    if response.error:
        print(f"SambaNova Generator error: {response.error}")
    else:
        print("[SambaNova] LLM output:")
        print(response.data)
