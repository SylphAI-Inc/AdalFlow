Multimodal Client Tutorial
=======================

This tutorial demonstrates how to use the OpenAI client for different types of tasks: text generation, vision analysis, and image generation.

Model Types
----------

The OpenAI client supports three types of operations:

1. Text/Chat Completion (``ModelType.LLM``)
   - Standard text generation
   - Vision analysis (with GPT-4V)
2. Image Generation (``ModelType.IMAGE_GENERATION``)
   - DALL-E image generation
3. Embeddings (``ModelType.EMBEDDER``)
   - Text embeddings

Basic Usage
----------

The model type is specified when creating a ``Generator`` instance:

.. code-block:: python

    from adalflow.core import Generator
    from adalflow.components.model_client.openai_client import OpenAIClient
    from adalflow.core.types import ModelType

    # Create the client
    client = OpenAIClient()

    # For text generation
    gen = Generator(
        model_client=client,
        model_kwargs={"model": "gpt-4", "max_tokens": 100},
        model_type=ModelType.LLM  # Specify LLM type
    )
    response = gen({"input_str": "Hello, world!"})

Vision Tasks
-----------

Vision tasks use ``ModelType.LLM`` since they are handled by GPT-4V:

.. code-block:: python

    # Vision analysis
    vision_gen = Generator(
        model_client=client,
        model_kwargs={
            "model": "gpt-4o-mini",
            "images": "path/to/image.jpg",
            "max_tokens": 300,
        },
        model_type=ModelType.LLM  # Vision uses LLM type
    )
    response = vision_gen({"input_str": "What do you see in this image?"})

Image Generation
--------------

For DALL-E image generation, use ``ModelType.IMAGE_GENERATION``:

.. code-block:: python

    # Image generation with DALL-E
    dalle_gen = Generator(
        model_client=client,
        model_kwargs={
            "model": "dall-e-3",
            "size": "1024x1024",
            "quality": "standard",
            "n": 1,
        },
        model_type=ModelType.IMAGE_GENERATION  # Specify image generation type
    )
    response = dalle_gen({"input_str": "A cat playing with yarn"})

Backward Compatibility
--------------------

For backward compatibility with existing code:

1. ``model_type`` defaults to ``ModelType.LLM`` if not specified
2. Older models that only support text continue to work with ``ModelType.LLM``
3. The OpenAI client handles the appropriate API endpoints based on the model type

Error Handling
-------------

The client includes error handling for:

1. Invalid model types for operations
2. Invalid image URLs or file paths
3. Unsupported model capabilities
4. API errors and rate limits

Complete Example
--------------

See the complete example in ``tutorials/multimodal_client_testing_examples.py``, which demonstrates:

1. Basic text generation
2. Vision analysis with image input
3. DALL-E image generation
4. Error handling for invalid inputs 