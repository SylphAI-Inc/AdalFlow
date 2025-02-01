import httpx
import logging

from typing import Any, Dict, Optional, Sequence

# AdalFlow imports -- adjust the import path if your repo structure differs
from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    ModelType,
    GeneratorOutput,
    EmbedderOutput,
    # Embedding,
    # Usage,
)

logger = logging.getLogger(__name__)


class XAIClient(ModelClient):
    """
    A ModelClient integration for xAI, making direct HTTP calls via httpx.

    Follows the AdalFlow ModelClient protocol:
    - init_sync_client
    - init_async_client
    - convert_inputs_to_api_kwargs
    - call (sync)
    - acall (async)
    - parse_chat_completion
    - parse_embedding_response (optional, if xAI or future endpoint supports embeddings)
    """

    BASE_URL = "https://api.x.ai/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        timeout: int = 30,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the xAI model client.

        Args:
            api_key (str): Your xAI API key.
            timeout (int): Default request timeout in seconds.
            base_url (str, optional): Override the default base URL if needed.
        """
        super().__init__()
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = base_url or self.BASE_URL

        # Initialize both sync and async clients
        self.sync_client = self.init_sync_client()
        self.async_client = self.init_async_client()

    def init_sync_client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout)

    def init_async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout)

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """
        Convert AdalFlow inputs into the xAI API's request payload.

        For LLM usage, we typically need:
          {
             "model": ...,
             "messages": [ { "role": "user", "content": "Hello" }, ... ],
             ...extra
          }

        Returns:
            Dict: A dictionary of arguments that `call()` or `acall()` can pass to httpx.
        """
        final_args = dict(model_kwargs)  # copy so we don’t mutate original

        if model_type == ModelType.LLM:
            messages = []

            if isinstance(input, str):
                messages = [{"role": "user", "content": input}]
            elif isinstance(input, Sequence):
                messages = input
            else:
                raise ValueError(
                    "For LLM usage, 'input' must be a string or list of messages."
                )

            final_args["messages"] = messages
            return final_args

        elif model_type == ModelType.EMBEDDER:
            # If xAI eventually supports embeddings, adapt as needed
            # For now, we either raise or do a placeholder
            if isinstance(input, str):
                input_list = [input]
            elif isinstance(input, Sequence):
                input_list = list(input)
            else:
                raise ValueError(
                    "For EMBEDDER usage, 'input' must be a string or list of strings."
                )

            final_args["input"] = input_list
            return final_args

        else:
            raise ValueError(f"model_type {model_type} is not supported by XAIClient.")

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        if not self.sync_client:
            raise RuntimeError("Synchronous client not initialized.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if model_type == ModelType.LLM:
            model = api_kwargs.get("model")
            if not model:
                raise ValueError("Missing 'model' in api_kwargs for xAI LLM call.")

            data = {
                "model": model,
                "messages": api_kwargs["messages"],
            }

            reserved_keys = {"model", "messages"}
            for k, v in api_kwargs.items():
                if k not in reserved_keys:
                    data[k] = v

            try:
                response = self.sync_client.post(
                    self.base_url,
                    headers=headers,
                    json=data,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as exc:
                logger.error(f"xAI sync call error: {exc}")
                raise exc

        elif model_type == ModelType.EMBEDDER:
            raise NotImplementedError("xAI embeddings not yet implemented.")

        else:
            raise ValueError(f"Unsupported model_type {model_type}")

    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        """
        Asynchronous call to xAI. We use httpx.AsyncClient to POST to the xAI endpoint.

        Args:
            api_kwargs (dict): The final request payload from convert_inputs_to_api_kwargs.
            model_type (ModelType): LLM or EMBEDDER.

        Returns:
            dict: The raw JSON response from xAI.
        """
        if not self.async_client:
            raise RuntimeError("Asynchronous client not initialized.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if model_type == ModelType.LLM:
            model = api_kwargs.get("model")
            if not model:
                raise ValueError("Missing 'model' in api_kwargs for xAI LLM call.")

            data = {
                "model": model,
                "messages": api_kwargs["messages"],
            }
            reserved_keys = {"model", "messages"}
            for k, v in api_kwargs.items():
                if k not in reserved_keys:
                    data[k] = v

            try:
                async with self.async_client as client:
                    response = await client.post(
                        self.base_url,
                        headers=headers,
                        json=data,
                    )
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPError as exc:
                logger.error(f"xAI async call error: {exc}")
                raise exc

        elif model_type == ModelType.EMBEDDER:
            raise NotImplementedError("xAI embeddings not yet implemented.")
        else:
            raise ValueError(f"Unsupported model_type {model_type}")

    def parse_chat_completion(self, completion: Dict) -> GeneratorOutput:
        """
        Convert the raw xAI chat response into a GeneratorOutput for AdalFlow.

        xAI’s typical JSON response might look something like:
            {
              "choices": [
                {
                  "message": {
                    "role": "assistant",
                    "content": "Hello there!"
                  }
                }
              ]
            }

        Returns:
            GeneratorOutput with the text from the first "assistant" message.
        """
        if not completion:
            return GeneratorOutput(
                data=None, error="Empty xAI response", raw_response=completion
            )

        try:
            content = completion["choices"][0]["message"]["content"]
            return GeneratorOutput(
                data=content,
                error=None,
                raw_response=completion,
            )
        except (KeyError, IndexError) as e:
            err_msg = f"Unexpected xAI response format: {e}"
            logger.error(err_msg)
            return GeneratorOutput(data=None, error=err_msg, raw_response=completion)

    def parse_embedding_response(self, response: Dict) -> EmbedderOutput:
        raise NotImplementedError("xAI embedding parse not yet implemented.")


if __name__ == "__main__":
    from adalflow.core import Generator
    from adalflow.utils import get_logger
    from dotenv import load_dotenv
    import os

    log = get_logger(level="DEBUG")
    load_dotenv()

    xai_api_key = os.getenv("XAI_API_KEY", "YOUR_XAI_API_KEY")

    xai_client = XAIClient(api_key=xai_api_key, timeout=30)

    generator = Generator(
        model_client=xai_client,
        model_kwargs={
            "model": "grok-2-latest",
            "temperature": 0,
            "stream": False,
        },
    )

    prompt_kwargs = {
        "input": [
            {"role": "system", "content": "You are a test assistant."},
            {
                "role": "user",
                "content": "Testing. Just say hi and hello world and nothing else.",
            },
        ]
    }

    response = generator(prompt_kwargs)

    if response.error:
        print(f"[xAI] Generator Error: {response.error}")
    else:
        print(f"[xAI] Response: {response.data}")
