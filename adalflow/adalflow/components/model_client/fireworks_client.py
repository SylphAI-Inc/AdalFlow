from typing import Optional, Any, Callable, Literal
from openai.types import Completion

from adalflow.components.model_client.openai_client import OpenAIClient

BASE_URL = "https://api.fireworks.ai/inference/v1/"


class FireworksClient(OpenAIClient):
    """
    A component wrapper for Fireworks AI's OpenAI-compatible API.

    Inherits from OpenAIClient but modifies:
      - base_url = "https://api.fireworks.ai/inference/v1/"
      - env_api_key_name = "FIREWORKS_API_KEY"
      - default input_type set to "messages" for multi-turn chat usage.

    Example usage with AdalFlow Generator:
      fw_client = FireworksClient()
      generator = Generator(model_client=fw_client, model_kwargs={"model": "llama-v3p1-8b-instruct"})
      result = generator({"input_str": "Hello from Fireworks AI!"})
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "messages",
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

    load_dotenv()
    log = get_logger(level="DEBUG")

    fw_client = FireworksClient()

    generator = Generator(
        model_client=fw_client,
        model_kwargs={
            "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "temperature": 0.7,
        },
    )

    prompt_kwargs = {
        "input_str": "Hello from Fireworks AI! Can you summarize the concept of quantum mechanics?"
    }

    response = generator(prompt_kwargs)

    if response.error:
        print(f"[Fireworks] Generator error: {response.error}")
    else:
        print("[Fireworks] LLM output:")
        print(response.data)
