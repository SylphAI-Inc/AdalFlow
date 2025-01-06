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

1. How to use OpenAI's multimodal capabilities in AdalFlow
2. Different ways to input images (local files, URLs)
3. Controlling image detail levels
4. Working with multiple images

Multimodal Support in OpenAIClient
--------------------------------

The :class:`OpenAIClient` supports both text and image inputs. For multimodal generation, you can use the following models:

- ``gpt-4o``: Versatile, high-intelligence flagship model
- ``gpt-4o-mini``: Fast, affordable small model for focused tasks (default)
- ``o1``: Reasoning model that excels at complex, multi-step tasks
- ``o1-mini``: Smaller reasoning model for complex tasks

The client supports:

- Local image files (automatically encoded to base64)
- Image URLs
- Multiple images in a single request
- Control over image detail level

Basic Usage
----------

First, install AdalFlow with OpenAI support:

.. code-block:: bash

    pip install "adalflow[openai]"

Then you can use the client with the Generator. By default, it uses ``gpt-4o-mini``, but you can specify any supported model:

.. code-block:: python

    from adalflow import Generator, OpenAIClient

    # Using the default gpt-4o-mini model
    generator = Generator(
        model_client=OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o-mini",  # or "gpt-4o", "o1", "o1-mini"
            "max_tokens": 300
        }
    )

    # Using an image URL
    response = generator(
        prompt="Describe this image.",
        images="https://example.com/image.jpg"
    )

    # Using the flagship model for more complex tasks
    generator_flagship = Generator(
        model_client=OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o",
            "max_tokens": 300
        }
    )

Image Detail Levels
-----------------

The client supports three detail levels:

- ``auto``: Let the model decide based on image size (default)
- ``low``: Low-resolution mode (512px x 512px)
- ``high``: High-resolution mode with detailed crops

.. code-block:: python

    generator = Generator(
        model_client=OpenAIClient(),
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
   - Model compatibility checking
   - Usage tracking

3. Output Format:
   - Returns standard :class:`GeneratorOutput` format
   - Includes model usage information
   - Preserves error messages if any occur

Limitations
---------

Be aware of these limitations when using multimodal features:

1. Model Support and Capabilities:
   - Four models available with different strengths:
     - ``gpt-4o``: Best for complex visual analysis and detailed understanding
     - ``gpt-4o-mini``: Good balance of speed and accuracy for common tasks
     - ``o1``: Excels at multi-step reasoning with visual inputs
     - ``o1-mini``: Efficient for focused visual reasoning tasks
   - The client will return an error if using an unsupported model with images

2. Image Size and Format:
   - Maximum file size: 20MB per image
   - Supported formats: PNG, JPEG, WEBP, non-animated GIF

3. Common Limitations:
   - May struggle with:
     - Very small or blurry text
     - Complex spatial relationships
     - Detailed technical diagrams
     - Non-Latin text or symbols

4. Cost and Performance Considerations:
   - Image inputs increase token usage
   - High detail mode uses more tokens
   - Consider using:
     - ``gpt-4o-mini`` for routine tasks
     - ``o1-mini`` for basic reasoning tasks
     - ``gpt-4o`` or ``o1`` for complex analysis

For more details, see the :class:`OpenAIClient` API reference.
