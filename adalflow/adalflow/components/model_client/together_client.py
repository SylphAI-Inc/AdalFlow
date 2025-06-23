import logging
import os
import re
from typing import Any, Optional, Callable, Literal

from adalflow.utils.lazy_import import safe_import, OptionalPackages

# Dynamically import the Together SDK components.
together = safe_import(
    OptionalPackages.TOGETHER.value[0], OptionalPackages.TOGETHER.value[1]
)

from together import Together, Completion, AsyncTogether
from adalflow.components.model_client.openai_client import OpenAIClient

logger = logging.getLogger(__name__)

__all__ = ["TogetherClient"]


class TogetherClient(OpenAIClient):
    __doc__ = r"""
    A minimal Together client that inherits from :class:`OpenAIClient`.

    This client is designed to work with the Together API by overriding the
    `init_sync_client` method from :class:`OpenAIClient`. It leverages the official
    Together SDK to initialize both synchronous and asynchronous client instances.

    References:
    - To get an API key, sign up at https://www.together.ai/
    - To list available models, use the command: `together models list`. For setup instructions, see:
        https://docs.together.ai/reference/installation

    **Example usage with the AdalFlow Generator:**

    .. code-block:: python

        from adalflow.core import Generator
        from adalflow.components.model_client.together_client import TogetherClient
        from adalflow.utils import setup_env

        setup_env()

        generator = Generator(
            model_client=TogetherClient(),
            model_kwargs={
                "model": "deepseek-ai/DeepSeek-R1",
                "temperature": 0.7,
                "max_tokens": 2000,
            },
            output_processors=your_output_processor_function,
        )

        prompt_kwargs = {"input_str": "Hi from AdalFlow! Summarize generative AI briefly."}

        response = generator(prompt_kwargs)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        non_streaming_chat_completion_parser: Callable[[Completion], Any] = None,
        streaming_chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: Optional[str] = None,
        env_api_key_name: str = "TOGETHER_API_KEY",
    ):
        """
        Initialize a TogetherClient instance.

        :param api_key: Together API key. If None, the client attempts to read from the environment variable ``TOGETHER_API_KEY``.
        :param chat_completion_parser: Optional function to parse responses from the Together API.
        :param input_type: The input format, either ``"text"`` or ``"messages"``. Defaults to ``"text"``.
        :param base_url: Optional API endpoint. Defaults to None.
        :param env_api_key_name: The name of the environment variable to use for the API key. Defaults to ``"TOGETHER_API_KEY"``.
        """
        super().__init__(
            api_key=api_key,
            non_streaming_chat_completion_parser=non_streaming_chat_completion_parser,
            streaming_chat_completion_parser=streaming_chat_completion_parser,
            input_type=input_type,
            base_url=base_url,
            env_api_key_name=env_api_key_name,
        )

    def init_sync_client(self):
        """
        Initialize and return a synchronous Together client instance.

        Retrieves the API key from the provided value or from the environment variable,
        and returns a Together client initialized with the proper base URL.
        """
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        return Together(api_key=api_key, base_url=self.base_url)

    def init_async_client(self):
        """
        Initialize and return an asynchronous Together client instance.

        If the Together SDK supports asynchronous operations, this method returns an
        instance of the async client.
        """
        return AsyncTogether(api_key=self._api_key, base_url=self.base_url)


if __name__ == "__main__":
    from adalflow.core import Generator
    from adalflow.utils import setup_env, get_logger
    from adalflow.core import func_to_data_component

    @func_to_data_component
    def extract_think_and_answer(text: str) -> Optional[dict]:
        """
        Extract text enclosed between <think>...</think> as 'think' and the text after </think> as 'answer'.

        Uses a regular expression to capture:
          - The content within the <think>...</think> tags.
          - The content that follows the </think> tag.

        :param text: The input string potentially containing <think> tags.
        :return: A dictionary with keys "think" and "answer" if a match is found, otherwise None.
        """
        pattern = r"<think>(.*?)</think>([\s\S]*)"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return {"think": match.group(1).strip(), "answer": match.group(2).strip()}
        return None

    # Set up logging and load environment variables.
    get_logger(enable_file=False)
    setup_env()

    # Instantiate the TogetherClient. The API key will be read from the environment if not provided.
    client = TogetherClient()

    # Create the Generator using the TogetherClient and specify model parameters.
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

    # Define the prompt to be processed.
    prompt_kwargs = {"input_str": "Hi from AdalFlow! Summarize generative AI briefly."}

    # Generate and output the response.
    response = generator(prompt_kwargs)

    if response.error:
        print(f"[TogetherSDK] Error: {response.error}")
    else:
        print(f"[TogetherSDK] Response: {response.data}")
