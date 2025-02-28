from typing import Optional, Any, Callable, Literal
from openai.types import Completion
from adalflow.components.model_client.openai_client import OpenAIClient

BASE_URL = "https://api.sambanova.ai/v1/"
__all__ = ["SambaNovaClient"]


class SambaNovaClient(OpenAIClient):
    __doc__ = r"""A component wrapper for SambaNova's OpenAI-compatible API.

This client extends :class:`OpenAIClient` and customizes:
  - The API key is read from the environment variable ``SAMBANOVA_API_KEY`` if not provided explicitly.

**Example usage with the AdalFlow Generator:**

.. code-block:: python

    from adalflow.core import Generator
    from adalflow.components.model_client.sambanova_client import SambaNovaClient

    generator = Generator(
        model_client=SambaNovaClient(),
        model_kwargs={
            "model": "Meta-Llama-3.1-8B-Instruct",
            "temperature": 0.7,
            "top_p": 0.9,
        },
    )

    prompt_kwargs = {
        "input_str": "Hello from SambaNova! Can you summarize the concept of quantum computing in simple terms?"
    }

    response = generator(prompt_kwargs)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        non_streaming_chat_completion_parser: Callable[[Completion], Any] = None,
        streaming_chat_completion_parser: Callable[[Completion], Any] = None,
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
            non_streaming_chat_completion_parser=non_streaming_chat_completion_parser,
            streaming_chat_completion_parser=streaming_chat_completion_parser,
            input_type=input_type,
            base_url=base_url,
            env_api_key_name=env_api_key_name,
        )
