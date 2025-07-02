from typing import Optional, Callable, Any, Literal
from openai.types import Completion
from adalflow.components.model_client.openai_client import OpenAIClient

BASE_URL = "https://api.fireworks.ai/inference/v1/"


class FireworksClient(OpenAIClient):
    __doc__ = r"""A component wrapper for Fireworks AI's OpenAI-compatible API.

    This class extends :class:`OpenAIClient` by customizing several key parameters:
    - Sets the API base URL to ``"https://api.fireworks.ai/inference/v1/"``.
    - Uses the environment variable ``"FIREWORKS_API_KEY"`` to obtain the API key.
    - Defaults the input type to ``"messages"``, which is suitable for multi-turn chat interactions.

    **Example usage with AdalFlow Generator:**

    .. code-block:: python

        from adalflow.core import Generator
        from adalflow.components.model_client.fireworks_client import FireworksClient


        generator = Generator(
            model_client=FireworksClient(),
            model_kwargs={
                "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
                "temperature": 0.7,
            }
        )

        prompt_kwargs = {
            "input_str": "Hello from Fireworks AI! Can you summarize the concept of quantum mechanics?"
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
        env_api_key_name: str = "FIREWORKS_API_KEY",
    ):
        super().__init__(
            api_key=api_key,
            non_streaming_chat_completion_parser=non_streaming_chat_completion_parser,
            streaming_chat_completion_parser=streaming_chat_completion_parser,
            input_type=input_type,
            base_url=base_url,
            env_api_key_name=env_api_key_name,
        )
