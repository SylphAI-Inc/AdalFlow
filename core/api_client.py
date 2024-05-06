r"""This is our attempt to componentize all API clients,  local or cloud.

For a particular API provider, such as OpenAI, we will have a class that inherits from APIClient.
It does four things:
(1) Initialize the client, sync and async.
(2) Convert the input to the API-specific format.
(3) Call the API with the right client api method.
(4) Handle API specific exceptions and errors to retry the call.
"""

import os
from typing import Any, Sequence, Dict


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

    def _call(self, kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        raise NotImplementedError(f"{type(self).__name__} must implement _call method")

    def _acall(self, **kwargs):
        pass

    def _combine_input_and_model_kwargs(
        self,
        input: Any,
        combined_model_kwargs: dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> dict:
        r"""
        Convert the Component's standard input and model_kwargs into API-specific format
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _combine_input_and_model_kwargs method"
        )

    def _track_usage(self, **kwargs):
        pass

    # TODO: implement the error process later. Customized error handling per api provider
    # def _backoff_on_exception(self, exception, max_time=5):
    #     return backoff.on_exception(
    #         backoff.expo,
    #         exception,
    #         max_time=max_time,
    #     )

    def call(
        self,
        *,
        input: Any,
        model_kwargs: dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Any:
        combined_model_kwargs = self._combine_input_and_model_kwargs(
            input, model_kwargs, model_type=model_type
        )
        return self._call(kwargs=combined_model_kwargs, model_type=model_type)

    def acall(
        self,
        *,
        input: Any,
        model_kwargs: dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Any:
        combined_model_kwargs = self._combine_input_and_model_kwargs(
            input, model_kwargs, model_type=model_type
        )
        return self._acall(**combined_model_kwargs)


# class OpenAIClientOld(Component):
#     def __init__(self):
#         super().__init__()
#         self.provider = "OpenAI"
#         self._init_sync_client()

#     def _init_sync_client(self):
#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             raise ValueError("Environment variable OPENAI_API_KEY must be set")
#         self.sync_client = OpenAI()

#     def _init_async_client(self):
#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             raise ValueError("Environment variable OPENAI_API_KEY must be set")
#         self.async_client = AsyncOpenAI()

#     def _call(self, **kwargs):
#         """
#         kwargs is the combined input and model_kwargs
#         """
#         raise NotImplementedError(f"{type(self).__name__} must implement _call method")

#     def _acall(self, **kwargs):
#         pass

#     def _combine_input_and_model_kwargs(
#         self,
#         input: Any,
#         combined_model_kwargs: dict = {},
#         model_type: ModelType = ModelType.UNDEFINED,
#     ) -> dict:
#         r"""
#         Convert the Component's standard input and model_kwargs into API-specific format
#         """
#         final_model_kwargs = combined_model_kwargs.copy()
#         if model_type == ModelType.EMBEDDER:
#             # convert input to input
#             assert isinstance(input, Sequence), "input must be a sequence of text"
#             final_model_kwargs["input"] = input
#         elif model_type == ModelType.LLM:
#             # convert input to messages
#             assert isinstance(input, Sequence), "input must be a sequence of messages"
#             final_model_kwargs["messages"] = input
#         else:
#             raise ValueError(f"model_type {model_type} is not supported")
#         return final_model_kwargs

#     def _track_usage(self, **kwargs):
#         """
#         Track usage of the API
#         """
#         pass

#     @backoff.on_exception(
#         backoff.expo,
#         (
#             APITimeoutError,
#             InternalServerError,
#             RateLimitError,
#             UnprocessableEntityError,
#             BadRequestError,
#         ),
#         max_time=5,
#     )
#     def call(
#         self,
#         *,
#         input: Any,
#         model_kwargs: dict = {},
#         model_type: ModelType = ModelType.LLM,
#     ) -> Any:

#         combined_model_kwargs = self._combine_input_and_model_kwargs(
#             input, model_kwargs, model_type=model_type
#         )
#         return self._call(**combined_model_kwargs)
