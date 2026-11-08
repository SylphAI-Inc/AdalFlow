from typing import Optional, Any, Callable, Literal
from openai.types import Completion
from adalflow.components.model_client.openai_client import OpenAIClient

BASE_URL = "https://api.runcrate.ai/v1"

__all__ = ["RuncrateClient"]


class RuncrateClient(OpenAIClient):

    __doc__ = r"""Runcrate's client is built on top of :class:`OpenAIClient` without overriding any methods.

Runcrate is an AI cloud with an OpenAI-compatible inference API for open-weight models, so only the
default ``base_url`` and the API key environment variable differ from :class:`OpenAIClient`.

References:
  - To obtain your API key, sign up at: https://runcrate.ai/
  - API documentation: https://runcrate.ai/docs/models/chat-completions
  - To list available models: ``GET https://api.runcrate.ai/v1/models``

**Example usage with the AdalFlow Generator:**

.. code-block:: python

    from adalflow.core import Generator
    from adalflow.components.model_client.runcrate_client import RuncrateClient
    from adalflow.utils import setup_env

    setup_env()  # reads RUNCRATE_API_KEY from your .env file or environment

    generator = Generator(
        model_client=RuncrateClient(),
        model_kwargs={
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "temperature": 0,
            "stream": False,
        }
    )

    output = generator(prompt_kwargs={"input_str": "Testing. Just say hi and hello world and nothing else."})
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        non_streaming_chat_completion_parser: Callable[[Completion], Any] = None,
        streaming_chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: str = BASE_URL,
        env_api_key_name: str = "RUNCRATE_API_KEY",
    ):
        super().__init__(
            api_key=api_key,
            non_streaming_chat_completion_parser=non_streaming_chat_completion_parser,
            streaming_chat_completion_parser=streaming_chat_completion_parser,
            input_type=input_type,
            base_url=base_url,
            env_api_key_name=env_api_key_name,
        )
