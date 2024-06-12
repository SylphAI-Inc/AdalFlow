Embedder
============
What you will learn?

1. What is ``Embedder`` and why is it designed this way?
2. When to use ``Embedder`` and how to use it?
3. How to batch processing with ``BatchEmbedder``?

:class:`core.embedder.Embedder` class is similar to ``Generator``, it is a user-facing component that orchestrates embedding models via ``ModelClient`` and ``output_processors``.
Compared with using ``ModelClient`` directly, ``Embedder`` further simplify the interface and output a standard ``EmbedderOutput`` format.

By switching the ``ModelClient``, you can use different embedding models in your task pipeline easily, or even embedd different data such as text, image, etc.

EmbedderOutput
--------------

:class:`core.types.EmbedderOutput` is a standard output format of ``Embedder``. It is a subclass of `DataClass` and it contains the following core fields:

- ``data``: a list of embeddings, each embedding if of type :class:`core.types.Embedding`.
- ``error``: Error message if any error occurs during the model inference stage. Failure in the output processing stage will raise an exception instead of setting this field.
- ``raw_response``: Used for failed model inference.

Additionally, we add three properties to the ``EmbedderOutput``:

- ``length``: The number of embeddings in the ``data``.
- ``embedding_dim``: The dimension of the embeddings in the ``data``.
- ``is_normalized``: Whether the embeddings are normalized to unit vector or not using ``numpy``.


Embedder in Action
-------------------
We currently support `all embedding models from OpenAI <https://platform.openai.com/docs/guides/embeddings>`_ and `'thenlper/gte-base' <https://huggingface.co/thenlper/gte-base>`_ from HuggingFace `transformers <https://huggingface.co/docs/transformers/en/index>`_.
We will use these two to demonstrate how to use ``Embedder``, one from the API provider and the other using local model. For the local model, you might need to ensure ``transformers`` is installed.

.. note ::
    The ``output_processors`` can be a component or a ``Sequential`` container to chain together multiple components. The output processors are applied in order and is adapted only on the ``data`` field of the ``EmbedderOutput``.

Use OpenAI API
^^^^^^^^^^^^^^^
Before you start ensure you config the API key either in the environment variable or `.env` file, or directly pass it to the ``OpenAIClient``.

.. code-block:: python

    from lightrag.core.embedder import Embedder
    from lightrag.components.model_client import OpenAIClient
    from lightrag.utils import setup_env # ensure you setup OPENAI_API_KEY in your project .env file

    model_kwargs = {
        "model": "text-embedding-3-small",
        "dimensions": 256,
        "encoding_format": "float",
    }

    query = "What is the capital of China?"

    queries = [query] * 100


    embedder = Embedder(model_client=OpenAIClient(), model_kwargs=model_kwargs)

We find the ``model_kwargs`` from the OpenAI API documentation. We setup `query` to demonstrate call on a single query and `queries` to demonstrate batch call.

**Visualize structure**: we use ``print(embedder)``. The output will be:

.. code-block:: 

    Embedder(
    model_kwargs={'model': 'text-embedding-3-small', 'dimensions': 256, 'encoding_format': 'float'}, 
    (model_client): OpenAIClient()
    )

**Embed single query**:
Run the embedder and print the length and embedding dimension of the output.

.. code-block:: python

    output = embedder(query)
    print(output.length, output.embedding_dim, output.is_normalized)
    # 1 256 True


**Embed batch queries**:

.. code-block:: python

    output = embedder(queries)
    print(output.length, output.embedding_dim)
    # 100 256

Use Local Model
^^^^^^^^^^^^^^^
Set up the embedder with the local model.

.. code-block:: python

    from lightrag.core.embedder import Embedder
    from lightrag.components.model_client import TransformersClient

    model_kwargs = {
        "model": "thenlper/gte-base",
        "device": "cpu",
    }

    model_kwargs = {"model": "thenlper/gte-base"}
    local_embedder = Embedder(model_client=TransformersClient(), model_kwargs=model_kwargs)

Now, call the embedder with the same query and queries.

.. code-block:: python

    output = local_embedder(query)
    print(output.length, output.embedding_dim, output.is_normalized)
    # 1 768 True

    output = local_embedder(queries)
    print(output.length, output.embedding_dim, output.is_normalized)
    # 100 768 True

Use Output Processors
^^^^^^^^^^^^^^^^^^^^^^^
If we want to decreate the embedding dimension to only 256 to save memory, we can customize an additional output processing step and pass it to embedder via the ``output_processors`` argument.

.. code-block:: python

    from lightrag.core.types import Embedding
    from lightrag.core.functional import normalize_vector
    from typing import List
    from lightrag.core.component import Component
    from copy import deepcopy
    
    class DecreaseEmbeddingDim(Component):
        def __init__(self, old_dim: int, new_dim: int,  normalize: bool = True):
            super().__init__()
            self.old_dim = old_dim
            self.new_dim = new_dim
            self.normalize = normalize
            assert self.new_dim < self.old_dim, "new_dim should be less than old_dim"

        def call(self, input: List[Embedding]) -> List[Embedding]:
            output: List[Embedding] = deepcopy(input)
            for embedding in output:
                old_embedding = embedding.embedding
                new_embedding = old_embedding[: self.new_dim]
                if self.normalize:
                    new_embedding = normalize_vector(new_embedding)
                embedding.embedding = new_embedding
            return output
        
        def _extra_repr(self) -> str:
            repr_str = f"old_dim={self.old_dim}, new_dim={self.new_dim}, normalize={self.normalize}"
            return repr_str

This output procesor will process on the ``data`` field of the ``EmbedderOutput``, which is of type ``List[Embedding]``. Thus we have ``input: List[Embedding] -> output: List[Embedding]`` in the ``call`` method.
Putting it all together, we can create a new embedder with the output processor.

.. code-block:: python

   local_embedder_256 = Embedder(
        model_client=TransformersClient(),
        model_kwargs=model_kwargs,
        output_processors=DecreaseEmbeddingDim(768, 256),
    )
    print(local_embedder_256)

The structure looks like:

.. code-block:: 

    Embedder(
    model_kwargs={'model': 'thenlper/gte-base'}, 
    (model_client): TransformersClient()
    (output_processors): DecreaseEmbeddingDim(old_dim=768, new_dim=256, normalize=True)
    )

Run a query:

.. code-block:: python

    output = local_embedder_256(query)
    print(output.length, output.embedding_dim, output.is_normalized)
    # 1 256 True

.. note::
    Please find relevant research on how directly decreasing the embedding dimension affects the performance of your downstream tasks. We simply used this as an example to demonstrate the output processor.

BatchEmbedder
--------------
Especially in data processing pipelines, you can often have more than 1000 queries to embed. We need to chunk our queries into smaller batches to avoid memory overflow.
:class:`core.embedder.BatchEmbedder` is designed to handle this situation. For now, the code is rather simple, but in the future it can be extended to support multi-processing when you use LightRAG in production data pipeline.

The BatchEmbedder orchestrates the ``Embedder`` and handles the batching process. To use it, you need to pass the ``Embedder`` and the batch size to the constructor.

.. code-block:: python

    from lightrag.core.embedder import BatchEmbedder
   
    batch_embedder = BatchEmbedder(embedder=local_embedder, batch_size=100)

    queries = [query] * 1000

    response = batch_embedder(queries)
    # 100%|██████████| 11/11 [00:04<00:00,  2.59it/s]


.. note::
    To integrate your own embedding model or from API providers, you need to implement your own subclass of ``ModelClient``.

.. admonition:: References
   :class: highlight

   - transformers: https://huggingface.co/docs/transformers/en/index
   - thenlper/gte-base model: https://huggingface.co/thenlper/gte-base


.. admonition:: API reference
   :class: highlight

   - :class:`core.embedder.Embedder`
   - :class:`core.embedder.BatchEmbedder`
   - :class:`core.types.EmbedderOutput`
   - :class:`core.types.Embedding`
   - :class:`components.model_client.openai_client.OpenAIClient`
   - :class:`components.model_client.transformers_client.TransformersClient`
   - :class:`core.functional.normalize_vector`