ModelClient
============
:ref:`ModelClient<core-model_client>` is the protocol and base class for all model inference SDKs (either via APIs or local) to communicate with LightRAG internal components/classes.
By switching off ``ModelClient``  in a ``Generator`` or ``Embedder`` component, you can make your prompt or ``Retriever`` model-agnostic.

Model Inference SDKs
-------------------
With cloud API providers like OpenAI, Groq, Anthropic, it often comes with a `sync` and an `async` client via their SDKs. 
For example:

.. code-block:: python

    from openai import OpenAI, AsyncOpenAI
    sync_client = OpenAI()
    async_client = AsyncOpenAI()

    # sync call using APIs 
    response = sync_client.chat.completions.create(...)

For local models, such as using `huggingface transformers`, you need to create this model inference SDKs yourself.
How you do this is highly flexible. Here is an example to use local embedding model (e.g. ``thenlper/gte-base``) as a model (Refer :class:`components.model_client.transformers_client.TransformerEmbedder` for details):

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





``ModelClient`` Protocol
-----------------------------------------------------------------------------------------------------------

We designed 5 abstract methods in the ``ModelClient`` class to be implemented by the subclass model type.
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
        input: API_INPUT_TYPE = None,
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
        input: API_INPUT_TYPE = None,  # user input
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
Typically an LLM needs to use `parse_chat_completion` to parse the completion to texts.

.. code-block:: python

    def parse_chat_completion(self, completion: Any) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} must implement parse_chat_completion method"
        )

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
