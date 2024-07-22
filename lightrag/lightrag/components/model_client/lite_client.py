import os
from typing import Dict, Optional, Any, List
import backoff
import logging
import warnings

from lightrag.utils.lazy_import import safe_import, OptionalPackages
from lightrag.core.model_client import ModelClient
from lightrag.core.types import EmbedderOutput


# Importation conditionnelle de litellm
litellm = safe_import(OptionalPackages.LITELLM.value[0], OptionalPackages.LITELLM.value[1])

from litellm import completion, embedding
from litellm.exceptions import (
    Timeout,
    InternalServerError,
    APIConnectionError,
    RateLimitError,
)

log = logging.getLogger(__name__)

class LiteClient(ModelClient):
    __doc__ = r"""A component wrapper for the LiteLM API client.

    Visit https://litelms.com/docs/ for more api details.
    Check https://litelms.com/docs/models for the available models.

    Tested LiteLM models: 07/19/2024
    - mixtral-8x7b
    - gemma-7b-it
    """
    
    def __init__(
        self,
        model: str = "llama3-8b-8192",
        messages: List = [],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 1,
        max_tokens: Optional[Dict[str, Any]] = None,
        **Kwargs: Any,
    ):
        super().__init__()
        self.model = model
        self.sync_client = self.init_sync_client()
        self.async_client = None  # only initialize if the async call is called

    def init_sync_client(self) -> Any:
        validate = litellm.validate_environment(self.model)
        if "keys_in_environment" in validate and not validate["keys_in_environment"]:
            ApiError = validate["missing_keys"]
            raise ValueError(f"Environment variable {ApiError} must be set")
        return completion(model=self.model)
    
    async def async_client(self) -> Any:
        validate = litellm.validate_environment(self.model)
        if "keys_in_environment" in validate and not validate["keys_in_environment"]:
            ApiError = validate["missing_keys"]
            raise ValueError(f"Environment variable {ApiError} must be set")
        return await completion(model=self.model)
    
    
    @backoff.on_exception(
        backoff.expo,
        (
            Timeout,
            APIConnectionError,
            InternalServerError,
            RateLimitError,
        ),
        max_time=5,
    )

    def parse_embedding_response(
        self,
        response: Dict[str, List[float]]
    ) -> EmbedderOutput:
        try:
            embeddings = embedding(model=self.model, input=response["input"])
            return EmbedderOutput(data=[embeddings])
        except Exception as e:
            log.error(f"Error parsing the embedding response: {e}")
            return EmbedderOutput(data=[], error=str(e), raw_response=response)
