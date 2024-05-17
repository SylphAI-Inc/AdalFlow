r"""This is our attempt to componentize all API clients,  local or cloud.

For a particular API provider, such as OpenAI, we will have a class that inherits from APIClient.
It does four things:
(1) Initialize the client, sync and async.
(2) Convert the input to the API-specific format.
(3) Call the API with the right client api method.
(4) Handle API specific exceptions and errors to retry the call.
"""

from typing import Any, Dict


from core.component import Component
from core.data_classes import ModelType


class APIClient(Component):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # take the *args and **kwargs to be compatible with the Component class
        # comvert args to attributes
        for i, arg in enumerate(args):
            super().__setattr__(f"arg_{i}", arg)
        # convert kwargs to attributes
        for key, value in kwargs.items():
            super().__setattr__(key, value)

        # TODO: recheck to see if we need to initialize the client here
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

    def _call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        r"""kwargs: all the arguments that the API call needs, subclass should implement this method.

        Additionally in subclass you can implement the error handling and retry logic here. See OpenAIClient for example.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _call method")

    async def _acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        r"""kwargs: all the arguments that the API async call needs, subclass should implement this method if the API supports async call"""
        pass

    def _combine_input_and_model_kwargs(
        self,
        input: Any,
        combined_model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        r"""
        Bridge the Component's standard input and model_kwargs into API-specific format
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _combine_input_and_model_kwargs method"
        )

    @staticmethod
    def _process_text(text: str) -> str:
        """
        This is specific to OpenAI API, as removing new lines could have better performance in the embedder
        """
        text = text.replace("\n", " ")
        return text

    # def format_input(self, *, input: Any) -> Any:
    #     """
    #     This is specific to APIClient.
    #     # convert your component input to the API-specific format
    #     """
    #     raise NotImplementedError(
    #         f"{type(self).__name__} must implement format_input method"
    #     )

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

    def call(
        self,
        input: Any,
        model_kwargs: dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Any:
        # adapt the format and the key for input and model_kwargs
        combined_model_kwargs = self._combine_input_and_model_kwargs(
            input, model_kwargs, model_type=model_type
        )
        return self._call(api_kwargs=combined_model_kwargs, model_type=model_type)

    async def acall(
        self,
        *,
        input: Any,
        model_kwargs: dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Any:
        combined_model_kwargs = self._combine_input_and_model_kwargs(
            input, model_kwargs, model_type=model_type
        )
        return await self._acall(
            api_kwargs=combined_model_kwargs, model_type=model_type
        )
