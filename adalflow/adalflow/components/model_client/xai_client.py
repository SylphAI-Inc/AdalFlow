from typing import Optional, Any, Callable, Literal
from openai.types import Completion
from adalflow.components.model_client.openai_client import OpenAIClient

BASE_URL = "https://api.x.ai/v1"

__all__ = ["XAIClient"]


class XAIClient(OpenAIClient):

    __doc__ = r"""xAI's client is built on top of :class:`OpenAIClient` without overriding any methods.

References:
  - To obtain your API key, sign up at: https://x.ai/
  - API documentation: https://docs.x.ai/docs/api-reference#list-models

**Example usage with the AdalFlow Generator:**

.. code-block:: python

    from adalflow.core import Generator
    from adalflow.components.model_client.xai_client import XAIClient
    from adalflow.utils import setup_env

    setup_env()

    generator = Generator(
        model_client=XAIClient(),
        model_kwargs={
            "model": "grok-2-latest",
            "temperature": 0,
            "stream": False,
        }
    )

    outpupt = generator(prompt_kwargs={"input_str": "Testing. Just say hi and hello world and nothing else."})
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        non_streaming_chat_completion_parser: Callable[[Completion], Any] = None,
        streaming_chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: str = BASE_URL,
        env_api_key_name: str = "XAI_API_KEY",
    ):
        super().__init__(
            api_key=api_key,
            non_streaming_chat_completion_parser=non_streaming_chat_completion_parser,
            streaming_chat_completion_parser=streaming_chat_completion_parser,
            input_type=input_type,
            base_url=base_url,
            env_api_key_name=env_api_key_name,
        )
