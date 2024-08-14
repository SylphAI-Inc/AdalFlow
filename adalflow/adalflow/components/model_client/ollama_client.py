"""Ollama ModelClient integration."""

import os
from typing import (
    Dict,
    Optional,
    Any,
    TypeVar,
    List,
    Type,
    Generator as GeneratorType,
    Union,
)
import backoff
import logging
import warnings
from adalflow.core.types import ModelType, GeneratorOutput

from adalflow.utils.lazy_import import safe_import, OptionalPackages

ollama = safe_import(OptionalPackages.OLLAMA.value[0], OptionalPackages.OLLAMA.value[1])
import ollama
from ollama import RequestError, ResponseError, GenerateResponse


from adalflow.core.model_client import ModelClient
from adalflow.core.types import EmbedderOutput, Embedding

log = logging.getLogger(__name__)

T = TypeVar("T")


def parse_stream_response(completion: GeneratorType) -> Any:
    """Parse the completion to a str. We use the generate with prompt instead of chat with messages."""
    for chunk in completion:
        log.debug(f"Raw chunk: {chunk}")
        raw_response = chunk["response"] if "response" in chunk else None
        yield GeneratorOutput(data=None, raw_response=raw_response)


def parse_generate_response(completion: GenerateResponse) -> "GeneratorOutput":
    """Parse the completion to a str. We use the generate with prompt instead of chat with messages."""
    if "response" in completion:
        log.debug(f"response: {completion}")
        raw_response = completion["response"]
        return GeneratorOutput(data=None, raw_response=raw_response)
    else:
        log.error(
            f"Error parsing the completion: {completion}, type: {type(completion)}"
        )
        # raise ValueError(
        #     f"Error parsing the completion: {completion}, type: {type(completion)}"
        # )
        return GeneratorOutput(
            data=None, error="Error parsing the completion", raw_response=completion
        )


class OllamaClient(ModelClient):
    __doc__ = r"""A component wrapper for the Ollama SDK client.

    To make a model work, you need to:

    - [Download Ollama app] Go to https://github.com/ollama/ollama?tab=readme-ov-file to download the Ollama app (command line tool).
      Choose the appropriate version for your operating system.

    - [Pull a model] Run the following command to pull a model:

    .. code-block:: shell

        ollama pull llama3

    - [Run a model] Run the following command to run a model:

    .. code-block:: shell

        ollama run llama3

    This model will be available at http://localhost:11434. You can also chat with the model at the terminal after running the command.

    Args:
        host (Optional[str], optional): Optional host URI.
            If not provided, it will look for OLLAMA_HOST env variable. Defaults to None.
            The default host is "http://localhost:11434".

    Setting model_kwargs:

        For LLM, expect model_kwargs to have the following keys:

        model (str, required):
            Use `ollama list` via your CLI or  visit ollama model page on https://ollama.com/library

        stream (bool, default: False ) – Whether to stream the results.


        options (Optional[dict], optional)
            Options that affect model output.

            # If not specified the following defaults will be assigned.

                "seed": 0, - Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt.

                "num_predict": 128, - Maximum number of tokens to predict when generating text. (-1  = infinite generation, -2 = fill context)

                "top_k": 40, - Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative.

                "top_p": 0.9, - 	Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.

                "tfs_z": 1, - Tail free sampling. This is used to reduce the impact of less probable tokens from the output. Disabled by default (e.g. 1) (More documentation here for specifics)

                "repeat_last_n": 64, - Sets how far back the model should look back to prevent repetition. (0 = disabled, -1 = num_ctx)

                "temperature": 0.8, - The temperature of the model. Increasing the temperature will make the model answer more creatively.

                "repeat_penalty": 1.1, - Sets how strongly to penalize repetitions. A higher value(e.g., 1.5 will penlaize repetitions more strongly, while lowe values *e.g., 0.9 will be more lenient.)

                "mirostat": 0.0, - Enable microstat smapling for controlling perplexity. (0 = disabled, 1 = microstat, 2 = microstat 2.0)

                "mirostat_tau": 0.5, - Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text.

                "mirostat_eta": 0.1, - Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive.

                "stop": ["\n", "user:"], - 	Sets the stop sequences to use. When this pattern is encountered the LLM will stop generating text and return. Multiple stop patterns may be set by specifying multiple separate stop parameters in a modelfile.

                "num_ctx": 2048, - Sets the size of the context window used to generate the next token.

        For EMBEDDER, expect model_kwargs to have the following keys:

        model (str, required):
            Use `ollama list` via your CLI or  visit ollama model page on https://ollama.com/library

        prompt (str, required):
            String that is sent to the Embedding model.

        options (Optional[dict], optional):
            See LLM args for defaults.

    References:

        - https://github.com/ollama/ollama-python
        - https://github.com/ollama/ollama
        - Models: https://ollama.com/library
        - Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
        - Options Parameters: https://github.com/ollama/ollama/blob/main/docs/modelfile.md.
        - LlamaCPP API documentation(Ollama is based on this): https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#low-level-api
        - LLM API: https://llama-cpp-python.readthedocs.io/en/stable/api-reference/#llama_cpp.Llama.create_completion

    Tested Ollama models: 7/9/24

    -  internlm2:latest
    -  llama3
    -  jina/jina-embeddings-v2-base-en:latest

    .. note::
       We use `embeddings` and `generate` apis from Ollama SDK.
       Please refer to https://github.com/ollama/ollama-python/blob/main/ollama/_client.py for model_kwargs details.
    """

    def __init__(self, host: Optional[str] = None):
        super().__init__()

        self._host = host or os.getenv("OLLAMA_HOST")
        if not self._host:
            warnings.warn(
                "Better to provide host or set OLLAMA_HOST env variable. We will use the default host http://localhost:11434 for now."
            )
            self._host = "http://localhost:11434"

        log.debug(f"Using host: {self._host}")

        self.init_sync_client()
        self.async_client = None  # only initialize if the async call is called

    def init_sync_client(self):
        """Create the synchronous client"""

        self.sync_client = ollama.Client(host=self._host)

    def init_async_client(self):
        """Create the asynchronous client"""

        self.async_client = ollama.AsyncClient(host=self._host)

    # NOTE: do not put yield and return in the same function, thus we separate the functions
    def parse_chat_completion(
        self, completion: Union[GenerateResponse, GeneratorType]
    ) -> "GeneratorOutput":
        """Parse the completion to a str. We use the generate with prompt instead of chat with messages."""
        log.debug(f"completion: {completion}, {isinstance(completion, GeneratorType)}")
        if isinstance(completion, GeneratorType):  # streaming
            return parse_stream_response(completion)
        else:
            return parse_generate_response(completion)

    def parse_embedding_response(
        self, response: Dict[str, List[float]]
    ) -> EmbedderOutput:
        r"""Parse the embedding response to a structure LightRAG components can understand.
        Pull the embedding from response['embedding'] and store it Embedding dataclass
        """
        try:
            embeddings = Embedding(embedding=response["embedding"], index=0)
            return EmbedderOutput(data=[embeddings])
        except Exception as e:
            log.error(f"Error parsing the embedding response: {e}")
            return EmbedderOutput(data=[], error=str(e), raw_response=response)

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        r"""Convert the input and model_kwargs to api_kwargs for the Ollama SDK client."""
        # TODO: ollama will support batch embedding in the future: https://ollama.com/blog/embedding-models
        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                final_model_kwargs["prompt"] = input
                return final_model_kwargs
            else:
                raise ValueError(
                    "Ollama does not support batch embedding yet. It only accepts a single string for input for now. Make sure you are not passing a list of strings"
                )
        elif model_type == ModelType.LLM:
            if input is not None and input != "":
                final_model_kwargs["prompt"] = input
                return final_model_kwargs
            else:
                raise ValueError("Input must be text")
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (RequestError, ResponseError),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        if "model" not in api_kwargs:
            raise ValueError("model must be specified")
        log.info(f"api_kwargs: {api_kwargs}")
        if not self.sync_client:
            self.init_sync_client()
        if self.sync_client is None:
            raise RuntimeError("Sync client is not initialized")
        if model_type == ModelType.EMBEDDER:
            return self.sync_client.embeddings(**api_kwargs)
        if model_type == ModelType.LLM:
            return self.sync_client.generate(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (RequestError, ResponseError),
        max_time=5,
    )
    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        if self.async_client is None:
            self.init_async_client()
        if self.async_client is None:
            raise RuntimeError("Async client is not initialized")
        if "model" not in api_kwargs:
            raise ValueError("model must be specified")
        if model_type == ModelType.EMBEDDER:
            return await self.async_client.embeddings(**api_kwargs)
        if model_type == ModelType.LLM:
            return await self.async_client.generate(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @classmethod
    def from_dict(cls: Type["OllamaClient"], data: Dict[str, Any]) -> "OllamaClient":
        obj = super().from_dict(data)
        # recreate the existing clients
        obj.sync_client = obj.init_sync_client()
        obj.async_client = obj.init_async_client()
        return obj

    def to_dict(self, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
        r"""Convert the component to a dictionary."""

        # combine the exclude list
        exclude = list(set(exclude or []) | {"sync_client", "async_client"})

        output = super().to_dict(exclude=exclude)
        return output


# TODO: add tests to stream and non stream case
# if __name__ == "__main__":
#     from adalflow.core.generator import Generator
#     from adalflow.components.model_client import OllamaClient, OpenAIClient
#     from adalflow.utils import setup_env, get_logger

#     log = get_logger(level="DEBUG")

#     setup_env()

#     ollama_ai = {
#         "model_client": OllamaClient(),
#         "model_kwargs": {
#             "model": "qwen2:0.5b",
#             "stream": True,
#         },
#     }
#     open_ai = {
#         "model_client": OpenAIClient(),
#         "model_kwargs": {
#             "model": "gpt-3.5-turbo",
#             "stream": False,
#         },
#     }
#     # generator = Generator(**open_ai)
#     # output = generator({"input_str": "What is the capital of France?"})
#     # print(output)

#     # generator = Generator(**ollama_ai)
#     # output = generator({"input_str": "What is the capital of France?"})
#     # print(output)
