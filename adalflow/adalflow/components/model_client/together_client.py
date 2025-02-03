"""Together Client is highly similar to OpenAIClient.
We inherited OpenAIClient and only need to override the init_sync_client method."""

import logging
import os
from typing import Any, Optional, Callable, Literal, Generator

from adalflow.utils.lazy_import import safe_import, OptionalPackages

together = safe_import(
    OptionalPackages.TOGETHER.value[0], OptionalPackages.TOGETHER.value[1]
)

from together import Together, Completion, AsyncTogether

from adalflow.components.model_client.openai_client import OpenAIClient


logger = logging.getLogger(__name__)


class TogetherClient(OpenAIClient):
    __doc__ = r"""Together Client is highly similar to OpenAIClient.
    We inherited OpenAIClient and only need to override the init_sync_client method.

    References:
    - To get the API key, sign up at https://www.together.ai/
    - To list models, use command: `together models list`. Setup here:https://docs.together.ai/reference/installation
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
        return AsyncTogether(api_key=self._api_key, base_url=self.base_url)


if __name__ == "__main__":
    from adalflow.core import Generator
    from adalflow.utils import setup_env, get_logger

    from adalflow.core import func_to_data_component
    import re

    @func_to_data_component
    def extract_think_and_answer(text: str) -> str:
        """
        Extracts text enclosed between <think>...</think> as 'think'
        and the text after </think> as 'answer'.

        Returns:
            dict: {
                "think": <content within <think>...</think>>,
                "answer": <content after </think>>
            }
            or None if no match is found.
        """

        # Use DOTALL so '.' will match newlines as well
        pattern = r"<think>(.*?)</think>([\s\S]*)"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return {"think": match.group(1).strip(), "answer": match.group(2).strip()}
        return None

    get_logger(enable_file=False)

    setup_env()

    client = TogetherClient()

    generator = Generator(
        model_client=client,
        model_kwargs={
            # "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "model": "deepseek-ai/DeepSeek-R1",
            "temperature": 0.7,
            "max_tokens": 2000,
        },
        output_processors=extract_think_and_answer,
    )

    prompt_kwargs = {"input_str": "Hi from AdalFlow! Summarize generative AI briefly."}

    response = generator(prompt_kwargs)

    if response.error:
        print(f"[TogetherSDK] Error: {response.error}")
    else:
        print(f"[TogetherSDK] Response: {response.data}")
