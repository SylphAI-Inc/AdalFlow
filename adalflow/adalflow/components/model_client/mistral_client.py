import os
import logging
from typing import Any, Callable, Dict, Optional

from mistralai import Mistral
from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    ModelType,
    CompletionUsage,
    GeneratorOutput,
    EmbedderOutput,
)

log = logging.getLogger(__name__)


class MistralClient(ModelClient):
    """
    A simple synchronous Mistral client without streaming.

    This class implements the minimal required methods for an AdalFlow ModelClient:
      1. __init__, which initializes the sync client.
      2. convert_inputs_to_api_kwargs, which converts AdalFlow inputs to Mistral's format.
      3. call, which calls Mistral's synchronous chat completion endpoint.
      4. parse_chat_completion, which parses the LLM output into a GeneratorOutput.
      5. track_completion_usage, which can attempt to read token usage (if Mistral returns it).
      6. parse_embedding_response, which we haven’t implemented (Mistral does not offer embeddings yet).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Any], Any] = None,
        input_type: str = "messages",
        env_api_key_name: str = "MISTRAL_API_KEY",
        **kwargs,
    ):
        """
        Initialize the Mistral API client (synchronously).

        :param api_key: Mistral API key (string). If None, reads from env variable MISTRAL_API_KEY.
        :param chat_completion_parser: A custom parser function if you wish to parse Mistral output differently.
        :param input_type: Whether your model input is "text" or "messages".
        :param env_api_key_name: Name of the environment variable with the API key.
        :param kwargs: Additional model config or other arguments (e.g., "model_kwargs").
        """
        super().__init__()

        self.api_key = api_key or os.getenv(env_api_key_name)
        if not self.api_key:
            raise ValueError(
                f"Mistral API key is missing. "
                f"Provide it to MistralClient(...) or set {env_api_key_name}."
            )

        self.chat_completion_parser = chat_completion_parser or self.default_parser
        self.input_type = input_type
        self.model_kwargs = kwargs.get("model_kwargs", {})
        self.sync_client = self.init_sync_client()

    def init_sync_client(self) -> Mistral:
        """Initialize a synchronous Mistral client from the official `mistralai` package."""
        return Mistral(api_key=self.api_key)

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """
        Convert AdalFlow inputs into the arguments that Mistral’s .chat.complete(...) expects.

        For an LLM: we supply a list of `messages=[{"role": "...", "content": "..."}]`.
        """
        if model_type != ModelType.LLM:
            raise ValueError(
                f"MistralClient only supports ModelType.LLM in this example. Got {
                    model_type}."
            )

        final_kwargs = dict(model_kwargs)

        if self.input_type == "messages":
            if isinstance(input, list):
                messages = input
            elif isinstance(input, str):
                messages = [{"role": "user", "content": input}]
            else:
                raise TypeError(
                    "Input must be str or list of {role, content} dictionaries."
                )
        elif self.input_type == "text":
            if not isinstance(input, str):
                raise TypeError("Input must be str when input_type == 'text'.")
            messages = [{"role": "user", "content": input}]
        else:
            raise ValueError(f"Unsupported input_type: {self.input_type}")

        final_kwargs["messages"] = messages
        final_kwargs["model"] = final_kwargs.get("model", "mistral-large-latest")

        return final_kwargs

    def call(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ) -> GeneratorOutput:
        """
        Make a synchronous chat-completion call to Mistral’s `chat.complete(...)`.
        Return a GeneratorOutput (the standard AdalFlow response for LLMs).
        """
        if model_type != ModelType.LLM:
            raise ValueError("MistralClient only supports LLM calls in this example.")

        try:
            response = self.sync_client.chat.complete(**api_kwargs)
            return self.parse_chat_completion(response)

        except Exception as e:
            log.error(f"Mistral call failed: {e}")
            raise

    def parse_chat_completion(self, completion: Any) -> GeneratorOutput:
        """
        Convert the raw Mistral completion into a standard GeneratorOutput.
        """
        try:
            content = completion.choices[0].message.content
        except Exception as exc:
            log.error(f"Error extracting content from Mistral completion: {exc}")
            return GeneratorOutput(data=None, error=str(exc), raw_response=completion)

        usage = self.track_completion_usage(completion)

        return GeneratorOutput(data=content, raw_response=completion, usage=usage)

    def track_completion_usage(self, completion: Any) -> CompletionUsage:
        """
        Attempt to parse Mistral’s token usage from `completion.usage`.
        If it’s missing, just return zero usage.
        """
        try:
            usage_obj = completion.usage
            return CompletionUsage(
                completion_tokens=usage_obj.completion_tokens,
                prompt_tokens=usage_obj.prompt_tokens,
                total_tokens=usage_obj.total_tokens,
            )
        except (AttributeError, TypeError):
            log.warning("No usage data in Mistral response.")
            return CompletionUsage(0, 0, 0)

    def parse_embedding_response(self, response: Any) -> EmbedderOutput:
        """
        Not implemented because Mistral doesn’t (yet) offer embeddings.
        """
        raise NotImplementedError("Embedding not supported by Mistral at this time.")

    @staticmethod
    def default_parser(completion: Any) -> str:
        """
        If you use a custom parser, you can pass it in the constructor.
        Otherwise, we just pick out the text from the first choice.
        """
        try:
            return completion.choices[0].message.content
        except Exception as e:
            log.error(f"Default parser failed: {e}")
            return ""


if __name__ == "__main__":
    from adalflow.core import Generator
    from adalflow.utils import get_logger
    from dotenv import load_dotenv

    log = get_logger(level="DEBUG")
    load_dotenv()

    mistral_client = MistralClient(
        api_key=os.getenv("MISTRAL_API_KEY"),
        input_type="text",
        model_kwargs={"model": "mistral-large-latest"},
    )

    generator = Generator(
        model_client=mistral_client,
        model_kwargs={"model": "mistral-large-latest", "temperature": 0.7},
    )

    prompt_kwargs = {"input_str": "Explain the importance of the Fibonacci sequence."}

    response = generator(prompt_kwargs)

    if response.error:
        print(f"Generator error: {response.error}")
    else:
        print(f"Mistral LLM output: {response.data}")
