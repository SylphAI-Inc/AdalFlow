from typing import Optional, Any, Callable, Literal
from openai.types import Completion
from adalflow.components.model_client.openai_client import OpenAIClient

BASE_URL = "https://api.minimax.io/v1"

__all__ = ["MiniMaxClient"]


class MiniMaxClient(OpenAIClient):

    __doc__ = r"""A component wrapper for MiniMax's OpenAI-compatible API.

MiniMax provides large language models accessible through an OpenAI-compatible API.
This client extends :class:`OpenAIClient` and customizes:
  - The base URL to ``https://api.minimax.io/v1``
  - The API key environment variable to ``MINIMAX_API_KEY``

Available models include ``MiniMax-M3`` (default; 512K context, up to 128K output, image input),
``MiniMax-M2.7``, and ``MiniMax-M2.7-highspeed``.

References:
  - To obtain your API key, sign up at: https://www.minimaxi.com/
  - API documentation: https://www.minimaxi.com/document/introduction

**Example usage with the AdalFlow Generator:**

.. code-block:: python

    from adalflow.core import Generator
    from adalflow.components.model_client.minimax_client import MiniMaxClient
    from adalflow.utils import setup_env

    setup_env()

    generator = Generator(
        model_client=MiniMaxClient(),
        model_kwargs={
            "model": "MiniMax-M3",
            "temperature": 0.7,
            "stream": False,
        }
    )

    output = generator(prompt_kwargs={"input_str": "Hello! Tell me about yourself."})
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        non_streaming_chat_completion_parser: Callable[[Completion], Any] = None,
        streaming_chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: str = BASE_URL,
        env_api_key_name: str = "MINIMAX_API_KEY",
    ):
        """
        Initialize a MiniMaxClient instance.

        :param api_key: (Optional) MiniMax API key. If not provided, the client
                        attempts to read from the environment variable ``MINIMAX_API_KEY``.
        :param non_streaming_chat_completion_parser: (Optional) A custom function to parse non-streaming responses.
        :param streaming_chat_completion_parser: (Optional) A custom function to parse streaming responses.
        :param input_type: Specifies the input format, either ``"text"`` or ``"messages"``.
                           Defaults to ``"text"``.
        :param base_url: MiniMax API endpoint. Defaults to ``"https://api.minimax.io/v1"``.
        :param env_api_key_name: The name of the environment variable holding the API key.
                                 Defaults to ``MINIMAX_API_KEY``.
        """
        super().__init__(
            api_key=api_key,
            non_streaming_chat_completion_parser=non_streaming_chat_completion_parser,
            streaming_chat_completion_parser=streaming_chat_completion_parser,
            input_type=input_type,
            base_url=base_url,
            env_api_key_name=env_api_key_name,
        )
