from typing import Optional, Any, Callable, Literal
from openai.types import Completion
from adalflow.components.model_client.openai_client import OpenAIClient

BASE_URL = "https://api.x.ai/v1"

__doc__ = r"""
xAI's client is built on top of :class:`OpenAIClient` without overriding any methods.
It is configured to work with x.ai's API by:
  - Setting the API endpoint to ``https://api.x.ai/v1``.
  - Reading the API key from the environment variable ``XAI_API_KEY`` if not provided.
  - Defaulting the input format to ``"messages"`` to support chat-based interactions.

References:
  - To obtain your API key, sign up at: https://x.ai/
  - API documentation: https://docs.x.ai/docs/api-reference#list-models

**Example usage with the AdalFlow Generator:**

.. code-block:: python

    from adalflow.core import Generator
    from adalflow.components.model_client.xai_client import XAIClient
    from adalflow.utils import setup_env, get_logger

    # Initialize logging and load environment variables.
    get_logger(enable_file=False)
    setup_env()

    # Instantiate the xAI client (the API key will be retrieved from XAI_API_KEY if not provided).
    client = XAIClient()

    # Create the Generator with xAI client and specify model parameters.
    generator = Generator(
        model_client=client,
        model_kwargs={
            "model": "grok-2-latest",
            "temperature": 0,
            "stream": False,
        }
    )

    # Define the conversation prompt.
    prompt_kwargs = {
        "input": [
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Testing. Just say hi and hello world and nothing else."},
        ]
    }

    # Generate the response and display the result.
    response = generator(prompt_kwargs)
    if response.error:
        print(f"[xAI] Generator Error: {response.error}")
    else:
        print(f"[xAI] Response: {response.data}")
"""


class XAIClient(OpenAIClient):
    r"""
    xAI's client is built on top of :class:`OpenAIClient` without any additional method overrides.

    This client is pre-configured to work with x.ai's API:
      - The API endpoint is set to ``https://api.x.ai/v1``.
      - The API key is obtained from the environment variable ``XAI_API_KEY`` if not provided.
      - The default input format is ``"messages"``, making it well-suited for chat-based interactions.

    For more details, see:
      - x.ai sign-up: https://x.ai/
      - API documentation: https://docs.x.ai/docs/api-reference#list-models

    See the module-level documentation for an extended example using the AdalFlow Generator.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: str = BASE_URL,
        env_api_key_name: str = "XAI_API_KEY",
    ):
        super().__init__(
            api_key=api_key,
            chat_completion_parser=chat_completion_parser,
            input_type=input_type,
            base_url=base_url,
            env_api_key_name=env_api_key_name,
        )


if __name__ == "__main__":
    from adalflow.core import Generator
    from adalflow.utils import setup_env, get_logger

    # Initialize logging and load environment variables.
    get_logger(enable_file=False)
    setup_env()

    # Instantiate the xAI client.
    client = XAIClient()

    # Set up the Generator with the xAI client and specify model parameters.
    generator = Generator(
        model_client=client,
        model_kwargs={"model": "grok-2-latest", "temperature": 0, "stream": False},
    )

    # Define the conversation prompt.
    prompt_kwargs = {
        "input": [
            {"role": "system", "content": "You are a test assistant."},
            {
                "role": "user",
                "content": "Testing. Just say hi and hello world and nothing else.",
            },
        ]
    }

    # Generate and display the response.
    response = generator(prompt_kwargs)
    if response.error:
        print(f"[xAI] Generator Error: {response.error}")
    else:
        print(f"[xAI] Response: {response.data}")
