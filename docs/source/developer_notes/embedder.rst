Embedder
============
What you will learn?

1. What is ``Embedder`` and why is it designed this way?
2. When to use ``Embedder`` and how to use it?

:class:`core.embedder.Embedder` class is similar to ``Generator``, it is a user-facing component that orchestrates embedding models via ``ModelClient`` and ``output_processors``.
Compared with using ``ModelClient`` directly, ``Embedder`` further simplify the interface and output a standard ``EmbedderOutput`` format.

By switching the ``ModelClient``, you can use different embedding models in your task pipeline easily, or even embedd different data such as text, image, etc.

EmbedderOutput
--------------

:class:`core.types.EmbedderOutput` is a standard output format of ``Embedder``. It is a subclass of `DataClass` and it contains the following core fields:

- ``data``: a list of embeddings, each embedding if of type :class:`core.types.Embedding`.
- ``error``: Error message if any error occurs during the model inference stage. Failure in the output processing stage will raise an exception instead of setting this field.
- ``raw_response``: Used for failed model inference.

You can use ``.length`` to get the number of embeddings in the ``data`` directly.

Embedder in Action
-------------------
We currently support `all embedding models from OpenAI <https://platform.openai.com/docs/guides/embeddings>`_ and `'thenlper/gte-base' <https://huggingface.co/thenlper/gte-base>`_ from HuggingFace `transformers <https://huggingface.co/docs/transformers/en/index>`_.



.. note::
    To integrate your own embedding model or from API providers, you need to implement your own subclass of ``ModelClient``.

.. admonition:: References
   :class: highlight

   - transformers: https://huggingface.co/docs/transformers/en/index
   - thenlper/gte-base model: https://huggingface.co/thenlper/gte-base


.. admonition:: API reference
   :class: highlight

   - :class:`core.embedder.Embedder`
   - :class:`core.types.EmbedderOutput`
   - :class:`core.types.Embedding`
   - :class:`components.model_client.openai_client.OpenAIClient`
   - :class:`components.model_client.transformers_client.TransformersClient`