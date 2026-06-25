"""TwelveLabs ModelClient integration.

TwelveLabs provides video-native foundation models:

- **Marengo** (``marengo3.0``): a multimodal embedding model that maps text,
  image, audio and video into the same 512-dim space. Used here as an
  ``Embedder`` backend (``ModelType.EMBEDDER``).
- **Pegasus** (``pegasus1.5``): a video-understanding model that generates
  text from a video. Used here as a ``Generator`` backend
  (``ModelType.LLM``) to analyze a video given a prompt.

Visit https://docs.twelvelabs.io/ for more API details. Get a free API key at
https://twelvelabs.io.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Sequence

import backoff

from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    ModelType,
    EmbedderOutput,
    Embedding,
    GeneratorOutput,
    CompletionUsage,
)

from adalflow.utils.lazy_import import safe_import, OptionalPackages

# optional import
twelvelabs = safe_import(
    OptionalPackages.TWELVELABS.value[0], OptionalPackages.TWELVELABS.value[1]
)

from twelvelabs import TwelveLabs, AsyncTwelveLabs
from twelvelabs.core.api_error import ApiError

log = logging.getLogger(__name__)

# Default models, overridable via ``model_kwargs["model"]``.
DEFAULT_EMBEDDER_MODEL = "marengo3.0"
DEFAULT_LLM_MODEL = "pegasus1.5"


class TwelveLabsClient(ModelClient):
    __doc__ = r"""A component wrapper for the TwelveLabs API client.

    Supports two model types:

    - ``ModelType.EMBEDDER`` -> Marengo multimodal embeddings (512-dim).
      The ``input`` is text (a single string or a list of strings).
    - ``ModelType.LLM`` -> Pegasus video understanding. The ``input`` is the
      prompt and ``model_kwargs`` must carry the video reference, e.g.
      ``model_kwargs={"model": "pegasus1.5", "video_url": "https://..."}`` or
      ``{"video_id": "<indexed-video-id>"}``.

    Visit https://docs.twelvelabs.io/ for more api details. Get a free API key
    at https://twelvelabs.io.

    .. note::
        For all ModelClient integration, such as TwelveLabsClient, if you want
        to subclass it, you need to import it from the module directly:

        ``from adalflow.components.model_client.twelvelabs_client import TwelveLabsClient``

        instead of using the lazy import:

        ``from adalflow.components.model_client import TwelveLabsClient``
    """

    def __init__(self, api_key: Optional[str] = None):
        r"""It is recommended to set the TWELVELABS_API_KEY environment variable
        instead of passing it as an argument.

        Args:
            api_key (Optional[str], optional): TwelveLabs API key. Defaults to None.
        """
        super().__init__()
        self._api_key = api_key
        self.sync_client = self.init_sync_client()
        self.async_client = None  # only initialize if the async call is called

    def init_sync_client(self):
        api_key = self._api_key or os.getenv("TWELVELABS_API_KEY")
        if not api_key:
            raise ValueError("Environment variable TWELVELABS_API_KEY must be set")
        return TwelveLabs(api_key=api_key)

    def init_async_client(self):
        api_key = self._api_key or os.getenv("TWELVELABS_API_KEY")
        if not api_key:
            raise ValueError("Environment variable TWELVELABS_API_KEY must be set")
        return AsyncTwelveLabs(api_key=api_key)

    # ---------------------------------------------------------------- parsing

    def parse_embedding_response(self, response: Any) -> EmbedderOutput:
        r"""Parse a list of Marengo embedding responses into an EmbedderOutput.

        ``response`` is the list of raw ``EmbeddingResponse`` objects returned
        by ``call`` (one per input string), since the API embeds a single text
        per request.
        """
        try:
            embeddings: List[Embedding] = []
            model: Optional[str] = None
            for index, item in enumerate(response):
                model = model or item.model_name
                segment = item.text_embedding.segments[0]
                embeddings.append(Embedding(embedding=segment.float_, index=index))
            return EmbedderOutput(data=embeddings, model=model, usage=None)
        except Exception as e:
            log.error(f"Error parsing embedding response: {e}")
            return EmbedderOutput(data=[], error=str(e), raw_response=response)

    def parse_chat_completion(self, completion: Any) -> GeneratorOutput:
        r"""Parse a Pegasus analyze response into a GeneratorOutput."""
        log.debug(f"completion: {completion}")
        try:
            data = completion.data
            usage = self.track_completion_usage(completion)
            return GeneratorOutput(
                data=None, usage=usage, raw_response=data, id=completion.id
            )
        except Exception as e:
            log.error(f"Error parsing completion: {e}")
            return GeneratorOutput(data=None, error=str(e), raw_response=completion)

    def track_completion_usage(self, completion: Any) -> CompletionUsage:
        usage = getattr(completion, "usage", None)
        if usage is None:
            return CompletionUsage()
        output_tokens = getattr(usage, "output_tokens", None)
        input_tokens = getattr(usage, "input_tokens", None)
        total = None
        if output_tokens is not None and input_tokens is not None:
            total = output_tokens + input_tokens
        return CompletionUsage(
            completion_tokens=output_tokens,
            prompt_tokens=input_tokens,
            total_tokens=total,
        )

    # --------------------------------------------------------- input mapping

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        r"""Bridge AdalFlow inputs into the per-call TwelveLabs api_kwargs.

        - EMBEDDER: ``input`` is a str or list of str. Returns
          ``{"model_name": ..., "texts": [...]}``.
        - LLM: ``input`` is the prompt str. The video reference comes from
          ``model_kwargs`` via ``video_url``, ``video_id`` or ``video_asset_id``.
          Returns the kwargs for ``client.analyze``.
        """
        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            if not isinstance(input, Sequence) or not all(
                isinstance(i, str) for i in input
            ):
                raise TypeError("input must be a string or a list of strings")
            model_name = final_model_kwargs.pop("model", DEFAULT_EMBEDDER_MODEL)
            return {"model_name": model_name, "texts": list(input)}

        elif model_type == ModelType.LLM:
            if input is not None and not isinstance(input, str):
                raise TypeError("input (the prompt) must be a string")
            model_name = final_model_kwargs.pop("model", DEFAULT_LLM_MODEL)
            api_kwargs: Dict[str, Any] = {"model_name": model_name}
            if input:
                api_kwargs["prompt"] = input

            # Resolve the video reference into the SDK's `video` context.
            video_url = final_model_kwargs.pop("video_url", None)
            video_id = final_model_kwargs.pop("video_id", None)
            video_asset_id = final_model_kwargs.pop("video_asset_id", None)
            if video_url is not None:
                api_kwargs["video"] = {"type": "url", "url": video_url}
            elif video_asset_id is not None:
                api_kwargs["video"] = {"type": "asset_id", "asset_id": video_asset_id}
            elif video_id is not None:
                api_kwargs["video_id"] = video_id
            else:
                raise ValueError(
                    "One of 'video_url', 'video_id' or 'video_asset_id' must be "
                    "provided in model_kwargs for Pegasus analyze"
                )

            api_kwargs.update(final_model_kwargs)  # max_tokens, temperature, ...
            return api_kwargs
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    # ---------------------------------------------------------------- calling

    @backoff.on_exception(backoff.expo, ApiError, max_time=5)
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        if model_type == ModelType.EMBEDDER:
            model_name = api_kwargs["model_name"]
            responses = [
                self.sync_client.embed.create(model_name=model_name, text=text)
                for text in api_kwargs["texts"]
            ]
            return responses
        elif model_type == ModelType.LLM:
            return self.sync_client.analyze(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(backoff.expo, ApiError, max_time=5)
    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        if self.async_client is None:
            self.async_client = self.init_async_client()
        if model_type == ModelType.EMBEDDER:
            model_name = api_kwargs["model_name"]
            responses = []
            for text in api_kwargs["texts"]:
                responses.append(
                    await self.async_client.embed.create(
                        model_name=model_name, text=text
                    )
                )
            return responses
        elif model_type == ModelType.LLM:
            return await self.async_client.analyze(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    # --------------------------------------------------------- serialization

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TwelveLabsClient":
        obj = super().from_dict(data)
        obj.sync_client = obj.init_sync_client()
        return obj

    def to_dict(self) -> Dict[str, Any]:
        r"""Convert the component to a dictionary."""
        exclude = ["sync_client", "async_client"]  # unserializable objects
        return super().to_dict(exclude=exclude)


if __name__ == "__main__":
    from adalflow.core import Embedder
    from adalflow.utils import setup_env

    setup_env()

    embedder = Embedder(
        model_client=TwelveLabsClient(),
        model_kwargs={"model": DEFAULT_EMBEDDER_MODEL},
    )
    output = embedder("A dog playing in the park")
    print(f"dim={len(output.data[0].embedding)}")
