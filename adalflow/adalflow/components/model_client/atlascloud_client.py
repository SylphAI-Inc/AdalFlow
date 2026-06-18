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


class AtlasCloudClient(OpenAIClient):
    """
    A component wrapper for the Atlas Cloud API client.

    Atlas Cloud (https://atlascloud.ai) provides a unified, OpenAI-compatible API
    serving 300+ open and proprietary models. Because the API mirrors OpenAI's chat
    completions interface, it can be used with OpenAI SDKs or any OpenAI-compatible
    software simply by adjusting the API base URL.

    This client extends `OpenAIClient` but modifies the default `base_url` to use
    Atlas Cloud's API endpoint.

    Documentation: https://docs.atlascloud.ai

    Args:
        api_key (Optional[str], optional): Atlas Cloud API key. Defaults to `None`.
            If not provided, it is read from the `ATLASCLOUD_API_KEY` environment variable.
        non_streaming_chat_completion_parser (Callable[[Completion], Any], optional): A function to parse API responses.
        streaming_chat_completion_parser (Callable[[Completion], Any], optional): A function to parse API responses.
        input_type (Literal["text", "messages"], optional): Defines how input is handled. Defaults to `"messages"`.
        base_url (str, optional): API base URL, defaults to `"https://api.atlascloud.ai/v1"`.
        env_api_key_name (str, optional): Environment variable name for the API key. Defaults to `"ATLASCLOUD_API_KEY"`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        non_streaming_chat_completion_parser: Callable[[Completion], Any] = None,
        streaming_chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "messages",
        base_url: str = "https://api.atlascloud.ai/v1",
        env_api_key_name: str = "ATLASCLOUD_API_KEY",
    ):
        """Initializes the Atlas Cloud API client with the correct base URL.

        The input_type is set to "messages" by default for compatibility with chat models.
        """
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
        model_client=AtlasCloudClient(),
        model_kwargs={"model": "deepseek-ai/deepseek-v4-pro", "stream": True},
    )

    gen_response = gen(prompt_kwargs)
    print(f"gen_response: {gen_response}")

    for genout in gen_response.data:
        print(f"genout: {genout}")
