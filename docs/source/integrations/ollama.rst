.. _ollama-integration:

Ollama Integration
==================

.. raw:: html

   <div style="display: flex; justify-content: flex-start; align-items: center; margin-bottom: 20px;">
      <a href="https://colab.research.google.com/github/SylphAI-Inc/AdalFlow/blob/main/tutorials/models/ollama_models.ipynb" target="_blank" style="margin-right: 10px;">
         <img alt="Try Ollama Integration in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align: middle;">
      </a>
      <a href="https://github.com/SylphAI-Inc/AdalFlow/blob/main/adalflow/tutorials/models/ollama_models.py" target="_blank" style="display: flex; align-items: center;">
         <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
         <span style="vertical-align: middle;"> Open Source Code</span>
      </a>
   </div>

AdalFlow provides comprehensive support for Ollama, enabling you to run open-source LLMs locally without depending on external APIs. This integration supports both synchronous and asynchronous operations, including streaming responses.

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

    # Install AdalFlow with Ollama support
    pip install adalflow[ollama]

    # Install Ollama (if not already installed)
    curl -fsSL https://ollama.com/install.sh | sh

    # Start Ollama server
    ollama serve

    # Pull a model (e.g., qwen2:0.5b, mistral, or gpt-oss)
    ollama pull qwen2:0.5b

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from adalflow.components.model_client import OllamaClient
    from adalflow.core import Generator

    # Initialize the Generator with OllamaClient
    generator = Generator(
        model_client=OllamaClient(host="http://localhost:11434"),
        model_kwargs={"model": "qwen2:0.5b"}
    )

    # Generate a response
    response = generator({"input_str": "Hello, what can you do?"})
    print(response.data)

Model Capabilities
------------------

Text Generation
~~~~~~~~~~~~~~~

Basic text generation with various open-source models:

.. code-block:: python

    # Configure with specific model parameters
    generator = Generator(
        model_client=OllamaClient(),
        model_kwargs={
            "model": "qwen2:0.5b",
            "options": {
                "temperature": 0.7,
                "num_predict": 512,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": 2048,
            }
        }
    )

    response = generator({"input_str": "Explain quantum computing"})
    print(response.data)

Streaming Responses
~~~~~~~~~~~~~~~~~~~

Real-time streaming for better user experience:

**Synchronous Streaming:**

.. code-block:: python

    # Enable streaming
    stream_generator = Generator(
        model_client=OllamaClient(),
        model_kwargs={
            "model": "qwen2:0.5b",
            "stream": True
        }
    )

    output = stream_generator.call(
        prompt_kwargs={"input_str": "Tell me a story"}
    )

    # Access the raw streaming response
    for chunk in output.raw_response:
        if "message" in chunk:
            print(chunk["message"]["content"], end='', flush=True)

**Asynchronous Streaming:**

.. code-block:: python

    import asyncio

    # Using async streaming
    output = await stream_generator.acall(
        prompt_kwargs={"input_str": "Tell me a story"}
    )

    # Access the raw async streaming response
    async for chunk in output.raw_response:
        if "message" in chunk:
            print(chunk["message"]["content"], end='', flush=True)

Chat vs Generate API
~~~~~~~~~~~~~~~~~~~~

Ollama supports two APIs for text generation:

.. code-block:: python

    # Chat API (default) - uses conversation format
    chat_generator = Generator(
        model_client=OllamaClient(),
        model_kwargs={"model": "qwen2:0.5b"}
    )

    # Generate API - uses raw prompt
    generate_generator = Generator(
        model_client=OllamaClient(),
        model_kwargs={
            "model": "qwen2:0.5b",
            "generate": True  # Use generate API instead of chat
        }
    )

Text Embeddings
~~~~~~~~~~~~~~~

Generate embeddings for semantic search and similarity:

.. code-block:: python

    from adalflow.core import Embedder

    embedder = Embedder(
        model_client=OllamaClient(),
        model_kwargs={"model": "nomic-embed-text"}
    )

    # Single text embedding
    text = "This is a sample text for embedding"
    embedding = embedder(input=text)
    print(f"Embedding dimension: {len(embedding.data[0].embedding)}")

Advanced Features
-----------------

Reasoning Models (GPT-OSS)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use OpenAI's GPT-OSS models locally with Ollama:

.. code-block:: python

    # Pull GPT-OSS model
    # Run in terminal: ollama pull gpt-oss:20b

    reasoning_gen = Generator(
        model_client=OllamaClient(),
        model_kwargs={
            "model": "gpt-oss:20b",
            "options": {
                "temperature": 0.7,
                "num_predict": 1024,
            }
        }
    )

    response = reasoning_gen({
        "input_str": "Solve this problem step by step: ..."
    })
    
    # Access reasoning process if available
    if response.thinking:
        print("Thinking:", response.thinking)
    print("Answer:", response.data)

Model Options
~~~~~~~~~~~~~

Complete list of configurable options:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Option
     - Default
     - Description
   * - ``seed``
     - 0
     - Random seed for reproducible generation
   * - ``num_predict``
     - 128
     - Maximum tokens to generate (-1 for infinite)
   * - ``temperature``
     - 0.8
     - Creativity level (0.0-2.0)
   * - ``top_k``
     - 40
     - Number of top tokens to consider
   * - ``top_p``
     - 0.9
     - Cumulative probability cutoff
   * - ``repeat_penalty``
     - 1.1
     - Penalty for repeated tokens
   * - ``num_ctx``
     - 2048
     - Context window size
   * - ``stop``
     - []
     - Stop sequences (e.g., ["\\n", "user:"])
   * - ``mirostat``
     - 0
     - Mirostat sampling (0=disabled, 1/2=enabled)

Available Models
----------------

Popular models compatible with Ollama:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Model
     - Size
     - Best For
   * - ``qwen2:0.5b``
     - 0.5B
     - Lightweight, fast inference, good for testing
   * - ``llama3``
     - 8B
     - General purpose, balanced performance
   * - ``mistral``
     - 7B
     - Fast inference, good for coding
   * - ``mixtral``
     - 8x7B
     - High quality, mixture of experts
   * - ``qwen2``
     - 0.5B-72B
     - Multilingual, various sizes
   * - ``codellama``
     - 7B-34B
     - Code generation and understanding
   * - ``gpt-oss``
     - 20B/120B
     - OpenAI's open-source reasoning model
   * - ``nomic-embed-text``
     - -
     - Text embeddings for semantic search

To see all available models, visit: https://ollama.com/library

Resources
---------

- `Ollama Documentation <https://github.com/ollama/ollama>`_
- `Ollama Model Library <https://ollama.com/library>`_
- `AdalFlow Documentation <https://adalflow.sylph.ai/>`_
- `Discord Community <https://discord.gg/ezzszrRZvT>`_
- `GitHub Issues <https://github.com/SylphAI-Inc/AdalFlow/issues>`_