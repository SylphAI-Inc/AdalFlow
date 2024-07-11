ModelClient
============

.. .. admonition:: Author
..    :class: highlight

..    `Li Yin <https://github.com/liyin2015>`_

.. What you will learn?

.. 1. What is ``ModelClient`` and why is it designed this way?
.. 2. How to intergrate your own ``ModelClient``?
.. 3. How to use ``ModelClient`` directly?


:ref:`ModelClient<core-model_client>` is the standardized protocol and base class for all model inference SDKs (either via APIs or local) to communicate with LightRAG internal components.
Therefore, by switching out the ``ModelClient`` in a ``Generator``, ``Embedder``, or ``Retriever`` (those components that take models), you can make these functional components model-agnostic.



.. figure:: /_static/images/model_client.png
    :align: center
    :alt: ModelClient
    :width: 400px

    The bridge between all model inference SDKs and internal components in LightRAG

.. note::

    All users are encouraged to customize their own ``ModelClient`` whenever needed. You can refer to our code in ``components.model_client`` directory.


Model Inference SDKs
------------------------

With cloud API providers like OpenAI, Groq, and Anthropic, it often comes with a `sync` and an `async` client via their SDKs.
For example:


.. code-block:: python

    from openai import OpenAI, AsyncOpenAI

    sync_client = OpenAI()
    async_client = AsyncOpenAI()

    # sync call using APIs
    response = sync_client.chat.completions.create(...)

For local models, such as using `huggingface transformers`, you need to create these model inference SDKs yourself.
How you do this is highly flexible.
Here is an example of using a local embedding model (e.g., ``thenlper/gte-base``) as a model (Refer to :class:`TransformerEmbedder<components.model_client.transformers_client.TransformerEmbedder>` for details).
It really is just normal model inference code.




ModelClient Protocol
-----------------------------------------------------------------------------------------------------------
A model client can be used to manage different types of models, we defined a :class:`ModelType<core.types.ModelType>` to categorize the model type.

.. code-block:: python

    class ModelType(Enum):
        EMBEDDER = auto()
        LLM = auto()
        RERANKER = auto()
        UNDEFINED = auto()

We designed 6 abstract methods in the `ModelClient` class that can be implemented by subclasses to integrate with different model inference SDKs.
We will use :class:`OpenAIClient<components.model_client.OpenAIClient>` as the cloud API example and :class:`TransformersClient<components.model_client.transformers_client.TransformersClient>` along with the local inference code :class:`TransformerEmbedder<components.model_client.transformers_client.TransformerEmbedder>` as an example for local model clients.


First, we offer two methods, `init_async_client` and `init_sync_client`, for subclasses to initialize the SDK client.
You can refer to :class:`OpenAIClient<components.model_client.OpenAIClient>` to see how these methods, along with the `__init__` method, are implemented:

This is how ``TransformerClient`` does the same thing:

.. code-block:: python

    class TransformersClient(ModelClient):
        def __init__(self) -> None:
            super().__init__()
            self.sync_client = self.init_sync_client()
            self.async_client = None
            support_model_list = {
                "thenlper/gte-base": {
                    "type": ModelType.EMBEDDER,
                }
            }

        def init_sync_client(self):
            return TransformerEmbedder()

Second, we use `convert_inputs_to_api_kwargs` for subclasses to convert LightRAG inputs into the `api_kwargs` (SDK arguments).

.. code-block:: python

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _combine_input_and_model_kwargs method"
        )

This is how `OpenAIClient` implements this method:

.. code-block:: python

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:

        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            # convert input to input
            assert isinstance(input, Sequence), "input must be a sequence of text"
            final_model_kwargs["input"] = input
        elif model_type == ModelType.LLM:
            messages: List[Dict[str, str]] = []
            if input is not None and input != "":
                messages.append({"role": "system", "content": input})
            assert isinstance(
                messages, Sequence
            ), "input must be a sequence of messages"
            final_model_kwargs["messages"] = messages
        else:
            raise ValueError(f"model_type {model_type} is not supported")
        return final_model_kwargs

.. For embedding, as `Embedder` takes both `str` and `List[str]` as input, we need to convert the input to a list of strings.
.. For LLM, as `Generator` takes a `prompt_kwargs` (dict) and converts it into a single string, we need to convert the input to a list of messages.
.. For Rerankers, you can refer to :class:`CohereAPIClient<components.model_client.cohere_client.CohereAPIClient>` for an example.


For embedding, as ``Embedder`` takes both `str` and `List[str]` as input, we need to convert the input to a list of strings that is acceptable by the SDK.
For LLM, as ``Generator`` will takes a `prompt_kwargs`(dict) and convert it into a single string, thus we need to convert the input to a list of messages.
For Rerankers, you can refer to :class:`CohereAPIClient<components.model_client.cohere_client.CohereAPIClient>` for an example.

This is how ``TransformerClient`` does the same thing:

.. code-block:: python

    def convert_inputs_to_api_kwargs(
            self,
            input: Any,
            model_kwargs: dict = {},
            model_type: ModelType = ModelType.UNDEFINED,
        ) -> dict:
            final_model_kwargs = model_kwargs.copy()
            if model_type == ModelType.EMBEDDER:
                final_model_kwargs["input"] = input
                return final_model_kwargs
            else:
                raise ValueError(f"model_type {model_type} is not supported")


In addition, you can add any method that parses the SDK-specific output to a format compatible with LightRAG components.
Typically, an LLM needs to use `parse_chat_completion` to parse the completion to text and `parse_embedding_response` to parse the embedding response to a structure that LightRAG components can understand.
You can refer to :class:`OpenAIClient<components.model_client.openai_client.OpenAIClient>` for API embedding model integration and :class:`TransformersClient<components.model_client.transformers_client.TransformersClient>` for local embedding model integration.


Lastly, the `call` and `acall` methods are used to call model inference via their own arguments.
We encourage subclasses to provide error handling and retry mechanisms in these methods.


The `OpenAIClient` example:

.. code-block:: python

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        if model_type == ModelType.EMBEDDER:
            return self.sync_client.embeddings.create(**api_kwargs)
        elif model_type == ModelType.LLM:
            return self.sync_client.chat.completions.create(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

The `TransformerClient` example:

.. code-block:: python

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
            return self.sync_client(**api_kwargs)

O
ur library currently integrates with six providers: OpenAI, Groq, Anthropic, Huggingface, Google, and Cohere.
Please check out :ref:`ModelClient Integration<components-model_client>`.



Use ModelClient directly
-----------------------------------------------------------------------------------------------------------


Though ``ModelClient`` is often managed in a ``Generator``, ``Embedder``, or ``Retriever`` component, you can use it directly if you plan to write your own component.
Here is an example of using ``OpenAIClient`` directly, first on an LLM model:


.. code-block:: python

    from lightrag.components.model_client import OpenAIClient
    from lightrag.core.types import ModelType
    from lightrag.utils import setup_env

    setup_env()

    openai_client = OpenAIClient()

    query = "What is the capital of France?"

    # try LLM model
    model_type = ModelType.LLM

    prompt = f"User: {query}\n"
    model_kwargs = {"model": "gpt-3.5-turbo", "temperature": 0.5, "max_tokens": 100}
    api_kwargs = openai_client.convert_inputs_to_api_kwargs(input=prompt,
                                                            model_kwargs=model_kwargs,
                                                            model_type=model_type)
    print(f"api_kwargs: {api_kwargs}")

    response = openai_client.call(api_kwargs=api_kwargs, model_type=model_type)
    response_text = openai_client.parse_chat_completion(response)
    print(f"response_text: {response_text}")

The output will be:

.. code-block::

    api_kwargs: {'model': 'gpt-3.5-turbo', 'temperature': 0.5, 'max_tokens': 100, 'messages': [{'role': 'system', 'content': 'User: What is the capital of France?\n'}]}
    response_text: The capital of France is Paris.

Then on Embedder model:

.. code-block:: python

    # try embedding model
    model_type = ModelType.EMBEDDER
    # do batch embedding
    input = [query] * 2
    model_kwargs = {"model": "text-embedding-3-small", "dimensions": 8, "encoding_format": "float"}
    api_kwargs = openai_client.convert_inputs_to_api_kwargs(input=input, model_kwargs=model_kwargs, model_type=model_type)
    print(f"api_kwargs: {api_kwargs}")



    response = openai_client.call(api_kwargs=api_kwargs, model_type=model_type)
    reponse_embedder_output = openai_client.parse_embedding_response(response)
    print(f"reponse_embedder_output: {reponse_embedder_output}")

The output will be:

.. code-block::

    api_kwargs: {'model': 'text-embedding-3-small', 'dimensions': 8, 'encoding_format': 'float', 'input': ['What is the capital of France?', 'What is the capital of France?']}
    reponse_embedder_output: EmbedderOutput(data=[Embedding(embedding=[0.6175549, 0.24047995, 0.4509756, 0.37041178, -0.33437008, -0.050995983, -0.24366009, 0.21549304], index=0), Embedding(embedding=[0.6175549, 0.24047995, 0.4509756, 0.37041178, -0.33437008, -0.050995983, -0.24366009, 0.21549304], index=1)], model='text-embedding-3-small', usage=Usage(prompt_tokens=14, total_tokens=14), error=None, raw_response=None)


.. TODO: add optional package introduction here


.. admonition:: API reference
   :class: highlight

   - :class:`core.model_client.ModelClient`
   - :class:`components.model_client.openai_client.OpenAIClient`
   - :class:`components.model_client.transformers_client.TransformersClient`
   - :class:`components.model_client.groq_client.GroqAPIClient`
   - :class:`components.model_client.anthropic_client.AnthropicAPIClient`
   - :class:`components.model_client.google_client.GoogleGenAIClient`
   - :class:`components.model_client.cohere_client.CohereAPIClient`
