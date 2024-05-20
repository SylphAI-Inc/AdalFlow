r"""This is our attempt to componentize all API clients,  local or cloud.

For a particular API provider, such as OpenAI, we will have a class that inherits from APIClient.
It does four things:
(1) Initialize the client, sync and async.
(2) Convert the input to the API-specific format.
(3) Call the API with the right client api method.
(4) Handle API specific exceptions and errors to retry the call.
"""

from typing import Any, Dict, Union, Optional


from core.component import Component
from core.data_classes import ModelType


class APIClient(Component):
    r"""Bridge the gap between LightRAG components and model APIs."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.sync_client = self._init_sync_client()
        self.async_client = None

    def _init_sync_client(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _init_sync_client method"
        )

    def _init_async_client(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _init_async_client method"
        )

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        r"""Subclass use this to call the API with the sync client.
        model_type: this decides which API, such as chat.completions or embeddings for OpenAI.
        api_kwargs: all the arguments that the API call needs, subclass should implement this method.

        Additionally in subclass you can implement the error handling and retry logic here. See OpenAIClient for example.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _call method")

    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        r"""Subclass use this to call the API with the async client."""
        pass

    def convert_input_to_api_kwargs(
        self,
        input: Any,  # user input
        system_input: Optional[
            Union[str]
        ] = None,  # system input that llm will use to generate the response
        combined_model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        r"""
        Bridge the Component's standard input and model_kwargs into API-specific format, the api_kwargs that will be used in _call and _acall methods.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _combine_input_and_model_kwargs method"
        )

    def parse_chat_completion(self, completion: Any) -> str:
        r"""
        Parse the chat completion to a structure your sytem standarizes. (here is str)
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement parse_chat_completion method"
        )

    @staticmethod
    def _process_text(text: str) -> str:
        """
        This is specific to OpenAI API, as removing new lines could have better performance in the embedder
        """
        text = text.replace("\n", " ")
        return text

    def _track_usage(self, **kwargs):
        pass

    # TODO: implement the error process later. Customized error handling per api provider
    # def _backoff_on_exception(self, exception, max_time=5):
    #     return backoff.on_exception(
    #         backoff.expo,
    #         exception,
    #         max_time=max_time,
    #     )

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
