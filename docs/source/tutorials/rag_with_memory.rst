.. _tutorials-rag_with_memory:

RAG with Memory
==============

This guide demonstrates how to implement a RAG system with conversation memory using AdalFlow, based on our `github_chat <https://github.com/SylphAI-Inc/github_chat>`_ reference implementation.

Overview
--------

The github_chat project is a practical RAG implementation that allows you to chat with GitHub repositories while maintaining conversation context. It demonstrates:

- Code-aware responses using RAG
- Memory management for conversation context
- Support for multiple programming languages
- Both web and command-line interfaces

Architecture
-----------

The system is built with several key components:

Data Pipeline
^^^^^^^^^^^^

.. code-block:: text

    Input Documents → Text Splitter → Embedder → Vector Database

The data pipeline processes repository content through:

1. Document reading and preprocessing
2. Text splitting for optimal chunk sizes
3. Embedding generation
4. Storage in vector database

RAG System
^^^^^^^^^^

.. code-block:: text

    User Query → RAG Component → [FAISS Retriever, Generator, Memory]
                      ↓
                  Response

The RAG system includes:

- FAISS-based retrieval for efficient similarity search
- LLM-based response generation
- Memory component for conversation history

Memory Management
---------------

The memory system maintains conversation context through:

1. Dialog turn tracking
2. Context preservation
3. Dynamic memory updates

This enables:

- Follow-up questions
- Reference to previous context
- More coherent conversations

Quick Start
----------

1. Installation:

.. code-block:: bash

    git clone https://github.com/SylphAI-Inc/github_chat
    cd github_chat
    poetry install

2. Set up your OpenAI API key:

.. code-block:: bash

    mkdir -p .streamlit
    echo 'OPENAI_API_KEY = "your-key-here"' > .streamlit/secrets.toml

3. Run the application:

.. code-block:: bash

    # Web interface
    poetry run streamlit run app.py

    # Repository analysis
    poetry run streamlit run app_repo.py

Example Usage
-----------

1. **Demo Version (app.py)**
   - Ask about Alice (software engineer)
   - Ask about Bob (data scientist)
   - Ask about the company cafeteria
   - Test memory with follow-up questions

2. **Repository Analysis (app_repo.py)**
   - Enter your repository path
   - Click "Load Repository"
   - Ask questions about classes, functions, or code structure
   - View implementation details in expandable sections

Implementation Details
-------------------

The system uses AdalFlow's components:

- :class:`core.embedder.Embedder` for document embedding
- :class:`core.retriever.Retriever` for similarity search
- :class:`core.generator.Generator` for response generation
- Custom memory management for conversation tracking

For detailed implementation examples, check out the `github_chat repository <https://github.com/SylphAI-Inc/github_chat>`_.
