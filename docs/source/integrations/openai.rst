.. _openai-integration:

OpenAI Integration
==================

.. raw:: html

   <div style="display: flex; justify-content: flex-start; align-items: center; margin-bottom: 20px;">
      <a href="https://colab.research.google.com/github/SylphAI-Inc/AdalFlow/blob/main/notebooks/integration/openai_integration.ipynb" target="_blank" style="margin-right: 10px;">
         <img alt="Try OpenAI Integration in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align: middle;">
      </a>
      <a href="https://github.com/SylphAI-Inc/AdalFlow/blob/main/tutorials/models/openai_models.py" target="_blank" style="display: flex; align-items: center;">
         <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
         <span style="vertical-align: middle;"> Open Source Code</span>
      </a>
   </div>

AdalFlow provides comprehensive support for OpenAI's models through the new **Response API** (``responses.create``), which unifies all model interactions including text generation, vision, reasoning, and image generation.

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

    pip install adalflow[openai]

Set up your API key from `platform.openai.com <https://platform.openai.com/api-keys>`_.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    import adalflow as adal
    from adalflow.components.model_client import OpenAIClient

    # Initialize the client - automatically uses the Response API
    generator = adal.Generator(
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o-mini"}
    )

    response = generator({"input_str": "Hello, world!"})
    print(response.data)

Response API Overview
---------------------

OpenAI's new Response API provides a unified interface for all model capabilities:

Key Features
~~~~~~~~~~~~

- **Unified Interface**: Single API endpoint for all model types
- **Typed Streaming**: Structured events (``ResponseTextDeltaEvent``, etc.)
- **Native Tools**: Built-in support for image generation
- **Simplified Multimodal**: Clean handling of text + image inputs

API Differences
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - Chat Completions API
     - Response API
   * - Endpoint
     - ``chat.completions.create``
     - ``responses.create``
   * - Input Format
     - ``messages``
     - ``input`` (string or messages)
   * - Multimodal
     - Complex content arrays
     - Simplified with ``images`` param
   * - Streaming
     - Untyped chunks
     - Typed events
   * - Tools
     - Function calling
     - Native tool types

Model Capabilities
------------------

Text Generation
~~~~~~~~~~~~~~~

Basic and streaming text generation with GPT models:

.. code-block:: python

    # Basic generation
    generator = adal.Generator(
        model_client=OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 150
        }
    )

    # Streaming
    from adalflow.components.model_client.utils import extract_text_from_response_stream

    streaming_gen = adal.Generator(
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "stream": True}
    )

    response = streaming_gen({"input_str": "Tell me a story"})
    for event in response.raw_response:
        text = extract_text_from_response_stream(event)
        if text:
            print(text, end="", flush=True)

Vision Models
~~~~~~~~~~~~~

Analyze images from URLs, local files, or base64 data:

.. code-block:: python

    # Single image
    vision_gen = adal.Generator(
        model_client=OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o",
            "images": "https://example.com/image.jpg"
        }
    )

    # Multiple images
    multi_gen = adal.Generator(
        model_client=OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o",
            "images": ["image1.jpg", "image2.jpg"]
        }
    )

    # Local file (auto-encoded to base64)
    local_gen = adal.Generator(
        model_client=OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o",
            "images": "/path/to/local/image.jpg"
        }
    )

Reasoning Models
~~~~~~~~~~~~~~~~

O1 and O1-mini models for complex problem solving:

.. code-block:: python

    reasoning_gen = adal.Generator(
        model_client=OpenAIClient(),
        model_kwargs={
            "model": "o1-mini",  # or "o1"
            "reasoning": {
                "effort": "medium",  # low, medium, high
                "summary": "auto"    # detailed, auto, none
            }
        }
    )

    response = reasoning_gen({"input_str": "Solve this complex problem..."})
    print(response.data)  # Solution
    print(response.thinking)  # Reasoning process

Image Generation
~~~~~~~~~~~~~~~~

Generate images using the new tools API:

.. code-block:: python

    image_gen = adal.Generator(
        model_client=OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o",
            "tools": [{"type": "image_generation"}]
        }
    )

    response = image_gen({
        "input_str": "Generate a sunset over mountains, watercolor style"
    })

    # Save generated images
    if response.images:
        saved_paths = response.save_images(
            directory="output",
            prefix="sunset",
            format="png"
        )
        print(f"Images saved to: {saved_paths}")

Mixed Generation
~~~~~~~~~~~~~~~~

Generate both text and images in one call:

.. code-block:: python

    mixed_gen = adal.Generator(
        model_client=OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o",
            "tools": [{"type": "image_generation"}]
        }
    )

    response = mixed_gen({
        "input_str": "Write a haiku and generate an image for it"
    })

    print(response.data)  # Haiku text
    if response.images:
        response.save_images(prefix="haiku")

Text Embeddings
~~~~~~~~~~~~~~~

Generate embeddings for semantic search:

.. code-block:: python

    from adalflow.core import Embedder

    embedder = Embedder(
        model_client=OpenAIClient(),
        model_kwargs={"model": "text-embedding-3-small"}
    )

    texts = ["text1", "text2", "text3"]
    embeddings = embedder(input=texts)

Helper Functions
----------------

The integration provides several helper functions for working with the Response API:

.. code-block:: python

    from adalflow.components.model_client.utils import (
        extract_text_from_response_stream,
        extract_complete_text_from_response_stream,
        is_response_complete,
        process_images_for_response_api,
        format_content_for_response_api
    )

Code Examples
-------------

Full working examples are available in:

- **Tutorial Script**: `tutorials/models/openai_models.py <https://github.com/SylphAI-Inc/AdalFlow/blob/main/adalflow/tutorials/models/openai_models.py>`_
- **Integration Notebook**: `notebooks/integration/openai_integration.ipynb <https://github.com/SylphAI-Inc/AdalFlow/blob/main/notebooks/integration/openai_integration.ipynb>`_

Source Code
-----------

- **OpenAI Client**: `openai_client.py <https://github.com/SylphAI-Inc/AdalFlow/blob/main/adalflow/adalflow/components/model_client/openai_client.py>`_
- **Response Utils**: `utils.py <https://github.com/SylphAI-Inc/AdalFlow/blob/main/adalflow/adalflow/components/model_client/utils.py>`_

Best Practices
--------------

1. **Image Handling**: URLs are most efficient; local files are auto-encoded to base64
2. **Streaming**: Use ``extract_text_from_response_stream()`` for text extraction
3. **Error Handling**: Always wrap API calls in try-except blocks
4. **Performance**: Set appropriate ``max_tokens`` and use streaming for better UX
5. **Monitoring**: Track usage with ``response.usage``

Resources
---------

- `OpenAI API Documentation <https://platform.openai.com/docs>`_
- `AdalFlow Documentation <https://adalflow.sylph.ai/>`_
- `Discord Community <https://discord.gg/ezzszrRZvT>`_
- `GitHub Issues <https://github.com/SylphAI-Inc/AdalFlow/issues>`_