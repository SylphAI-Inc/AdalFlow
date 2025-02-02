from typing import Optional, Any, Callable, Literal
from openai.types import Completion

from adalflow.components.model_client.openai_client import OpenAIClient

BASE_URL = "https://api.sambanova.ai/v1/"


class SambaNovaClient(OpenAIClient):
    """
    A component wrapper for SambaNova's OpenAI-compatible API.

    Inherits from OpenAIClient, but modifies:
      - base_url = "https://api.sambanova.ai/v1/"
      - env_api_key_name = "SAMBANOVA_API_KEY"
      - default input_type set to "messages" for chat usage.

    Example usage with AdalFlow Generator:
      sn_client = SambaNovaClient()
      generator = Generator(model_client=sn_client, model_kwargs={"model": "Meta-Llama-3.1-8B-Instruct"})
      output = generator({"input_str": "Hello from SambaNova!"})
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "messages",
        base_url: str = BASE_URL,
        env_api_key_name: str = "SAMBANOVA_API_KEY",
    ):
        """
        :param api_key: (optional) SambaNova API key; else read from SAMBANOVA_API_KEY env variable
        :param chat_completion_parser: (optional) custom parsing function for SambaNova responses
        :param input_type: "messages" or "text". Typically "messages" for multi-turn chat usage.
        :param base_url: SambaNova endpoint. Defaults to "https://api.sambanova.ai/v1/"
        :param env_api_key_name: Env var name for your key. Defaults to "SAMBANOVA_API_KEY"
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

    load_dotenv()

    log = get_logger(level="DEBUG")

    sn_client = SambaNovaClient()

    generator = Generator(
        model_client=sn_client,
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

    if response.error:
        print(f"SambaNova Generator error: {response.error}")
    else:
        print("[SambaNova] LLM output:")
        print(response.data)
