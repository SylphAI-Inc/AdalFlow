.. _tutorials-multimodal:

Multimodal Generation
===================

.. raw:: html

   <div style="display: flex; justify-content: flex-start; align-items: center; margin-bottom: 20px;">
      <a href="https://colab.research.google.com/github/SylphAI-Inc/AdalFlow/blob/main/notebooks/tutorials/adalflow_multimodal.ipynb" target="_blank" style="margin-right: 10px;">
         <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align: middle;">
      </a>
      <a href="https://github.com/SylphAI-Inc/AdalFlow/blob/main/notebooks/tutorials/adalflow_multimodal.ipynb" target="_blank" style="display: flex; align-items: center;">
         <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
         <span style="vertical-align: middle;"> View Source</span>
      </a>
   </div>

What you will learn?
------------------

1. How to use the OpenAI multimodal client for image understanding
2. Different ways to input images (local files, URLs)
3. Controlling image detail levels
4. Working with multiple images

The OpenAIMultimodalClient
------------------------

The :class:`OpenAIMultimodalClient` extends AdalFlow's model client capabilities to handle images along with text. It supports:

- Local image files (automatically encoded to base64)
- Image URLs
- Multiple images in a single request
- Control over image detail level

Basic Usage
----------

First, install AdalFlow with OpenAI support:

.. code-block:: bash

    pip install "adalflow[openai]"

Then you can use the client with the Generator:

.. code-block:: python

    from adalflow import Generator, OpenAIMultimodalClient

    generator = Generator(
        model_client=OpenAIMultimodalClient(),
        model_kwargs={
            "model": "gpt-4o-mini",
            "max_tokens": 300
        }
    )

    # Using an image URL
    response = generator(
        prompt="Describe this image.",
        images="https://example.com/image.jpg"
    )

Image Detail Levels
-----------------

The client supports three detail levels:

- ``auto``: Let the model decide based on image size (default)
- ``low``: Low-resolution mode (512px x 512px)
- ``high``: High-resolution mode with detailed crops

.. code-block:: python

    generator = Generator(
        model_client=OpenAIMultimodalClient(),
        model_kwargs={
            "model": "gpt-4o-mini",
            "detail": "high"  # or "low" or "auto"
        }
    )

Multiple Images
-------------

You can analyze multiple images in one request:

.. code-block:: python

    images = [
        "path/to/local/image.jpg",
        "https://example.com/image.jpg"
    ]

    response = generator(
        prompt="Compare these images.",
        images=images
    )

Implementation Details
-------------------

The client handles:

1. Image Processing:
   - Automatic base64 encoding for local files
   - URL validation and formatting
   - Detail level configuration

2. API Integration:
   - Proper message formatting for OpenAI's vision models
   - Error handling and response parsing
   - Usage tracking

3. Output Format:
   - Returns standard :class:`GeneratorOutput` format
   - Includes model usage information
   - Preserves error messages if any occur

Limitations
---------

Be aware of these limitations when using the multimodal client:

1. Image Size:
   - Maximum file size: 20MB per image
   - Supported formats: PNG, JPEG, WEBP, non-animated GIF

2. Model Capabilities:
   - Best for general visual understanding
   - May struggle with:
     - Small text
     - Precise spatial relationships
     - Complex graphs
     - Non-Latin text

3. Cost Considerations:
   - Image inputs are metered in tokens
   - High detail mode uses more tokens
   - Consider using low detail mode for cost efficiency

For more details, see the :class:`OpenAIMultimodalClient` API reference.
