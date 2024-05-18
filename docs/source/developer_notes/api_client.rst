abstract class APIClient
============
This abstract class is to define ways for each API provider to communicate with LightRAG components. This
includes cloud API provider and local API provider where you can wrap a `huggingface transformer model` as an API client.

Most APIs are provided with `Sync` and `Async` clients. Each will be used in `call`(synchronous) and `acall`(asynchronous) methods. 
We have seen how different API providers process the user query and the system message differently. Thus for the `APIClient` class we designed
five major methods that the subclass will have to implement besides of the `__init__` method.

Private methods to initialize the client.
------------------------------------------
.. code-block:: python

    def _init_sync_client(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _init_sync_client method"
        )

    def _init_async_client(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _init_async_client method"
        )

Public methods used in `Model` components such as `Generator` and `Embedder` to communicate with the API.
-----------------------------------------------------------------------------------------------------------
For the subclass model type to convert their input into the `api_kwargs` that can be used in `call` and `acall` methods,

.. code-block:: python

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

The `call` method is used to make a synchronous call to the API. The subclass model type will have to implement this method.

.. code-block:: python

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

**See how each API provider has implemented these classes in the `components.api_client` section.**