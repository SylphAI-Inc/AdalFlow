"""Cohere ModelClient integration."""

import os
from typing import Dict, Optional, Any
import backoff
from lightrag.utils.lazy_import import safe_import, OptionalPackages

cohere = safe_import(OptionalPackages.COHERE.value[0], OptionalPackages.COHERE.value[1])
from cohere import (
    BadRequestError,
    InternalServerError,
)


from lightrag.core.model_client import ModelClient
from lightrag.core.types import ModelType


class CohereAPIClient(ModelClient):
    __doc__ = r"""A component wrapper for the Cohere API.

    Visit https://docs.cohere.com/ for more api details.

    References:
    - Cohere reranker: https://docs.cohere.com/reference/rerank

    Tested Cohere models: 6/16/2024
    -  rerank-english-v3.0, rerank-multilingual-v3.0, rerank-english-v2.0, rerank-multilingual-v2.0

    .. note::
        For all ModelClient integration, such as CohereAPIClient, if you want to subclass CohereAPIClient, you need to import it from the module directly.

        ``from lightrag.components.model_client.cohere_client import CohereAPIClient``

        instead of using the lazy import with:

        ``from lightrag.components.model_client import CohereAPIClient``
    """

    def __init__(self, api_key: Optional[str] = None):
        r"""It is recommended to set the GROQ_API_KEY environment variable instead of passing it as an argument.

        Args:
            api_key (Optional[str], optional): Groq API key. Defaults to None.
        """
        super().__init__()
        self._api_key = api_key
        self.init_sync_client()

        self.async_client = None  # only initialize if the async call is called

    def init_sync_client(self):
        api_key = self._api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("Environment variable COHERE_API_KEY must be set")
        self.sync_client = cohere.Client(api_key=api_key)

    def init_async_client(self):
        api_key = self._api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("Environment variable COHERE_API_KEY must be set")
        self.async_client = cohere.AsyncClient(api_key=api_key)

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,  # for retriever, it is a list of string.
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        r"""
        For rerank model, expect model_kwargs to have the following keys:
         model: str,
         query: str,
         documents: List[str],
         top_n: int,
        """
        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.RERANKER:
            final_model_kwargs["query"] = input
            assert "model" in final_model_kwargs, "model must be specified"
            assert "documents" in final_model_kwargs, "documents must be specified"
            assert "top_k" in final_model_kwargs, "top_k must be specified"

            # convert top_k to the api specific, which is top_n
            final_model_kwargs["top_n"] = final_model_kwargs.pop("top_k")
            return final_model_kwargs
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (
            BadRequestError,
            InternalServerError,
        ),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        assert (
            "model" in api_kwargs
        ), f"model must be specified in api_kwargs: {api_kwargs}"
        if (
            model_type == ModelType.RERANKER
        ):  # query -> # scores for top_k documents, index for the top_k documents, return as tuple

            response = self.sync_client.rerank(**api_kwargs)
            top_k_scores = [result.relevance_score for result in response.results]
            top_k_indices = [result.index for result in response.results]
            return top_k_indices, top_k_scores
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (
            BadRequestError,
            InternalServerError,
        ),
        max_time=5,
    )
    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        if self.async_client is None:
            self.init_async_client()
        assert "model" in api_kwargs, "model must be specified"
        if model_type == ModelType.RERANKER:
            response = await self.async_client.rerank(**api_kwargs)
            top_k_scores = [result.relevance_score for result in response.results]
            top_k_indices = [result.index for result in response.results]
            return top_k_indices, top_k_scores
        else:
            raise ValueError(f"model_type {model_type} is not supported")
