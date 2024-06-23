ModelClient
============
What you will learn?

1. What is ``ModelClient`` and why is it designed this way?
2. How to intergrate your own ``ModelClient``?
3. How to use ``ModelClient`` directly?

:ref:`ModelClient<core-model_client>` is the standardized protocol and base class for all model inference SDKs (either via APIs or local) to communicate with LightRAG internal components/classes.
Because so, by switching off ``ModelClient``  in a ``Generator`` or ``Embedder`` component, you can make your prompt or ``Retriever`` model-agnostic.


.. figure:: /_static/images/model_client.png
    :align: center
    :alt: ModelClient
    :width: 400px

    The interface to internal components in LightRAG

.. note::

    All users are encouraged to customize your own ``ModelClient`` whenever you need to do so. You can refer our code in ``components.model_client`` dir.

Model Inference SDKs
------------------------
With cloud API providers like OpenAI, Groq, Anthropic, it often comes with a `sync` and an `async` client via their SDKs. 
For example:

.. code-block:: python

    from openai import OpenAI, AsyncOpenAI

    sync_client = OpenAI()
    async_client = AsyncOpenAI()

    # sync call using APIs 
    response = sync_client.chat.completions.create(...)

For local models, such as using `huggingface transformers`, you need to create this model inference SDKs yourself.
How you do this is highly flexible. Here is an example to use local embedding model (e.g. ``thenlper/gte-base``) as a model (Refer :class:`components.model_client.transformers_client.TransformerEmbedder` for details).
It really is just normal model inference code.

.. code-block:: python

    from transformers import AutoTokenizer, AutoModel

    class TransformerEmbedder:
        models: Dict[str, type] = {}

        def __init__(self, model_name: Optional[str] = "thenlper/gte-base"):
            super().__init__()

            if model_name is not None:
                self.init_model(model_name=model_name)

        @lru_cache(None)
        def init_model(self, model_name: str):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                # register the model
                self.models[model_name] = self.model

            except Exception as e:
                log.error(f"Error loading model {model_name}: {e}")
                raise e

        def infer_gte_base_embedding(
            self,
            input=Union[str, List[str]],
            tolist: bool = True,
        ):
            model = self.models.get("thenlper/gte-base", None)
            if model is None:
                # initialize the model
                self.init_model("thenlper/gte-base")

            if isinstance(input, str):
                input = [input]
            # Tokenize the input texts
            batch_dict = self.tokenizer(
                input, max_length=512, padding=True, truncation=True, return_tensors="pt"
            )
            outputs = model(**batch_dict)
            embeddings = average_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )
            # (Optionally) normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            if tolist:
                embeddings = embeddings.tolist()
            return embeddings

        def __call__(self, **kwargs):
            if "model" not in kwargs:
                raise ValueError("model is required")
            # load files and models, cache it for the next inference
            model_name = kwargs["model"]
            # inference the model
            if model_name == "thenlper/gte-base":
                return self.infer_gte_base_embedding(kwargs["input"])
            else:
                raise ValueError(f"model {model_name} is not supported")





ModelClient Protocol
-----------------------------------------------------------------------------------------------------------
A model client can be used to manage different types of models, we defined a ``ModelType`` to categorize the model type.

.. code-block:: python

    class ModelType(Enum):
        EMBEDDER = auto()
        LLM = auto()
        UNDEFINED = auto()

We designed 6 abstract methods in the ``ModelClient`` class to be implemented by the subclass model type.
We will use :class:`components.model_client.OpenAIClient` along with the above ``TransformerEmbedder`` as examples.

First, we offer two methods to initialize the model SDKs:

.. code-block:: python

    def init_sync_client(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _init_sync_client method"
        )

    def init_async_client(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _init_async_client method"
        )

This is how `OpenAIClient` implements these methods along with ``__init__`` method:

.. code-block:: python

    class OpenAIClient(ModelClient):

        def __init__(self, api_key: Optional[str] = None):
  
            super().__init__()
            self._api_key = api_key
            self.sync_client = self.init_sync_client()
            self.async_client = None  # only initialize if the async call is called

        def init_sync_client(self):
            api_key = self._api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Environment variable OPENAI_API_KEY must be set")
            return OpenAI(api_key=api_key)

        def init_async_client(self):
            api_key = self._api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Environment variable OPENAI_API_KEY must be set")
            return AsyncOpenAI(api_key=api_key)

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


Second. we use `convert_inputs_to_api_kwargs` for subclass to convert LightRAG inputs into the `api_kwargs` (SDKs arguments).

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

In addition, you can add any method that parse the SDK specific output to a format compatible with LightRAG components.
Typically an LLM needs to use `parse_chat_completion` to parse the completion to texts and `parse_embedding_response` to parse the embedding response to a structure LightRAG components can understand.


.. code-block:: python

    def parse_chat_completion(self, completion: Any) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} must implement parse_chat_completion method"
        )

    def parse_embedding_response(self, response: Any) -> EmbedderOutput:
    r"""Parse the embedding response to a structure LightRAG components can understand."""
    raise NotImplementedError(
        f"{type(self).__name__} must implement parse_embedding_response method"
    )

You can refer to :class:`components.model_client.openai_client.OpenAIClient` for API embedding model integration and :class:`components.model_client.transformers_client.TransformersClient` for local embedding model integration.

Then `call` and `acall` methods to call Model inference via their own arguments.
We encourage the subclass provides error handling and retry mechanism in these methods.

.. code-block:: python

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        raise NotImplementedError(f"{type(self).__name__} must implement _call method")

    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        pass

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


Our library currently integrated with 5 providers: OpenAI, Groq, Anthropic, Huggingface, and Google.
Please check out :ref:`ModelClient Integration<components-model_client>`.

Use ModelClient directly
-----------------------------------------------------------------------------------------------------------
Though ``ModelClient`` is often managed in a ``Generator`` or ``Embedder`` component, you can use it directly if you ever plan to write your own component.
Here is an example to use ``OpenAIClient`` directly, first on LLM model:

.. code-block:: python

    from lightrag.components.model_client import OpenAIClient
    from lightrag.core.types import ModelType
    from lightrag.utils import setup_env

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

.. admonition:: API reference
   :class: highlight

   - :class:`core.model_client.ModelClient`
   - :class:`components.model_client.openai_client.OpenAIClient`
   - :class:`components.model_client.transformers_client.TransformersClient`
   - :class:`components.model_client.groq_client.GroqAPIClient`
   - :class:`components.model_client.anthropic_client.AnthropicAPIClient`
   - :class:`components.model_client.google_client.GoogleGenAIClient`
