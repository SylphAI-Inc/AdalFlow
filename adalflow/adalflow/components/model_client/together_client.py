"""Together Client is highly similar to OpenAIClient.
We inherited OpenAIClient and only need to override the init_sync_client method."""

import logging
import os
from typing import Any, Optional, Callable, Literal, Generator


# together = safe_import(
#     OptionalPackages.TOGETHER.value[0], OptionalPackages.TOGETHER.value[1]
# )

from together import Together, Completion

from adalflow.components.model_client.openai_client import OpenAIClient


logger = logging.getLogger(__name__)


class TogetherClient(OpenAIClient):
    __doc__ = r"""Together Client is highly similar to OpenAIClient.
    We inherited OpenAIClient and only need to override the init_sync_client method

    References:
    - To get the API key, sign up at https://www.together.ai/
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: Optional[str] = None,
        env_api_key_name: str = "TOGETHER_API_KEY",
    ):
        super().__init__(
            api_key=api_key,
            chat_completion_parser=chat_completion_parser,
            input_type=input_type,
            base_url=base_url,
            env_api_key_name=env_api_key_name,
        )

    def init_sync_client(self):
        """
        For the official Together library, we don't strictly need a separate sync client.
        We'll just return None or we could return self.together_client if we want.
        """
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        return Together(api_key=api_key, base_url=self.base_url)

    def init_async_client(self):
        """
        If Together offers an async interface, we could store that here. If not, just None.
        """
        return None


if __name__ == "__main__":
    from adalflow.core import Generator
    from adalflow.utils import setup_env, get_logger

    get_logger(enable_file=False)

    setup_env()

    client = TogetherClient()

    generator = Generator(
        model_client=client,
        model_kwargs={
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "temperature": 0.7,
            "max_tokens": 64,
        },
    )

    prompt_kwargs = {
        "input": "Hi from AdalFlow! Summarize generative AI briefly."
        # or
        # "input": [
        #     {"role":"system","content":"You are a helpful assistant."},
        #     {"role":"user","content":"What's new in generative AI?"}
        # ]
    }

    response = generator(prompt_kwargs)

    if response.error:
        print(f"[TogetherSDK] Error: {response.error}")
    else:
        print(f"[TogetherSDK] Response: {response.data}")
