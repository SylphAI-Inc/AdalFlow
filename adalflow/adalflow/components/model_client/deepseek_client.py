from typing import (
    Optional,
    Any,
    Callable,
    Literal,
)

from adalflow.utils.lazy_import import safe_import, OptionalPackages
from adalflow.components.model_client.openai_client import OpenAIClient
from openai.types import Completion

openai = safe_import(OptionalPackages.OPENAI.value[0], OptionalPackages.OPENAI.value[1])


class DeepSeekClient(OpenAIClient):
    """
    A component wrapper for the DeepSeek API client.

    DeepSeek's API is compatible with OpenAI's API, making it possible to use OpenAI SDKs
    or OpenAI-compatible software with DeepSeek by adjusting the API base URL.

    This client extends `OpenAIClient` but modifies the default `base_url` to use DeepSeek's API.

    Documentation: https://api-docs.deepseek.com/guides/reasoning_model

    Args:
        api_key (Optional[str], optional): DeepSeek API key. Defaults to `None`.
        non_streaming_chat_completion_parser (Callable[[Completion], Any], optional): A function to parse API responses.
        streaming_chat_completion_parser (Callable[[Completion], Any], optional): A function to parse API responses.
        input_type (Literal["text", "messages"], optional): Defines how input is handled. Defaults to `"text"`.
        base_url (str, optional): API base URL, defaults to `"https://api.deepseek.com/v1/"`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        non_streaming_chat_completion_parser: Callable[[Completion], Any] = None,
        streaming_chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "messages",
        base_url: str = "https://api.deepseek.com/v1/",
        env_api_key_name: str = "DEEPSEEK_API_KEY",
    ):
        """Initializes DeepSeek API client with the correct base URL. The input_type is set to "messages" by default to be compatible with DeepSeek reasoner."""
        super().__init__(
            api_key=api_key,
            non_streaming_chat_completion_parser=non_streaming_chat_completion_parser,
            streaming_chat_completion_parser=streaming_chat_completion_parser,
            input_type=input_type,
            base_url=base_url,
            env_api_key_name=env_api_key_name,
        )


# Example usage:
if __name__ == "__main__":
    from adalflow.core import Generator
    from adalflow.utils import setup_env, get_logger

    log = get_logger(level="DEBUG")

    prompt_kwargs = {"input_str": "What is the meaning of life?"}

    setup_env()

    gen = Generator(
        model_client=DeepSeekClient(),
        model_kwargs={"model": "deepseek-reasoner", "stream": True},
    )

    gen_response = gen(prompt_kwargs)
    print(f"gen_response: {gen_response}")

    for genout in gen_response.data:
        print(f"genout: {genout}")
