.. raw:: html

   <div style="display: flex; justify-content: flex-start; align-items: center; margin-bottom: 20px;">
      <a href="https://colab.research.google.com/github/SylphAI-Inc/AdalFlow/blob/main/notebooks/tutorials/adalflow_rag_documents.ipynb" target="_blank" style="margin-right: 20px;">
         <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="height: 20px;">
      </a>

      <a href="https://github.com/SylphAI-Inc/AdalFlow/tree/main/tutorials/adalflow_rag_documents.py" target="_blank" style="display: flex; align-items: center;">
         <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; width: 20px; margin-right: 5px;">
         <span style="vertical-align: middle;"> Open Source Code</span>
      </a>
   </div>

RAG for documents
=============================

Overview
--------

This implementation showcases an end-to-end RAG system capable of handling large-scale text files and
generating context-aware responses. It is both modular and extensible, making it adaptable to various
use cases and LLM APIs.

**Imports**

- **SentenceTransformer**: Used for creating dense vector embeddings for textual data.
- **FAISS**: Provides efficient similarity search using vector indexing.
- **tiktoken**: ensures that the text preprocessing aligns with the tokenization requirements of the underlying language models, making the pipeline robust and efficient.
- **GroqAPIClient and OpenAIClient**: Custom classes for interacting with different LLM providers.
- **ModelType**: Enum for specifying the model type.

.. code-block:: python

    import os
    import tiktoken
    from typing import List, Dict, Tuple
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from faiss import IndexFlatL2

    from adalflow.components.model_client import GroqAPIClient, OpenAIClient
    from adalflow.core.types import ModelType
    from adalflow.utils import setup_env

The ``AdalflowRAGPipeline`` class sets up the Retrieval-Augmented Generation (RAG) pipeline. Its ``__init__`` method initializes key components:

- An embedding model (``all-MiniLM-L6-v2``) is loaded using ``SentenceTransformer`` to convert text into dense vector embeddings with a dimensionality of 384.
- A FAISS index (``IndexFlatL2``) is created for similarity-based document retrieval.
- Parameters such as ``top_k_retrieval`` (number of documents to retrieve) and ``max_context_tokens`` (limit on token count in the context) are configured.
- A tokenizer (``tiktoken``) ensures precise token counting, crucial for handling large language models (LLMs).

The method also initializes storage for documents, their embeddings, and associated metadata for efficient management and retrieval.

The ``AdalflowRAGPipeline`` class provides a flexible pipeline for Retrieval-Augmented Generation (RAG),
initializing with parameters such as the embedding model (``all-MiniLM-L6-v2`` by default), vector dimension,
top-k retrieval count, and token limits for context. It utilizes a tokenizer for token counting, a
SentenceTransformer for embeddings, and a FAISS index for similarity searches, while also maintaining
document data and metadata. The ``load_text_file`` method processes large text files into manageable chunks
by splitting the content into fixed line groups, facilitating easier embedding and storage. To handle
multiple files, ``add_documents_from_directory`` iterates over text files in a directory, embeds the content,
and stores them in the FAISS index along with metadata. Token counting is achieved via the ``count_tokens``
method, leveraging a tokenizer to precisely determine the number of tokens in a given text. The
``retrieve_and_truncate_context`` method fetches the most relevant documents from the FAISS index based on
query embeddings, truncating the context to adhere to token limits. Finally, the ``generate_response`` method
constructs a comprehensive prompt by combining the retrieved context and query, invokes the provided model
client for a response, and parses the results into a readable format. This pipeline demonstrates seamless
integration of text retrieval and generation to handle large-scale document queries effectively.


.. code-block:: python

    class AdalflowRAGPipeline:
        def __init__(self,
                    model_client=None,
                    model_kwargs=None,
                    embedding_model='all-MiniLM-L6-v2',
                    vector_dim=384,
                    top_k_retrieval=3,
                    max_context_tokens=800):
            """
            Initialize RAG Pipeline for handling large text files

            Args:
                embedding_model (str): Sentence transformer model for embeddings
                vector_dim (int): Dimension of embedding vectors
                top_k_retrieval (int): Number of documents to retrieve
                max_context_tokens (int): Maximum tokens to send to LLM
            """
            # Initialize model client for generation
            self.model_client = model_client

            # Initialize tokenizer for precise token counting
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

            # Initialize embedding model
            self.embedding_model = SentenceTransformer(embedding_model)

            # Initialize FAISS index for vector similarity search
            self.index = IndexFlatL2(vector_dim)

            # Store document texts, embeddings, and metadata
            self.documents = []
            self.document_embeddings = []
            self.document_metadata = []

            # Retrieval and context management parameters
            self.top_k_retrieval = top_k_retrieval
            self.max_context_tokens = max_context_tokens

            # Model generation parameters
            self.model_kwargs = model_kwargs

        def load_text_file(self, file_path: str) -> List[str]:
            """
            Load a large text file and split into manageable chunks

            Args:
                file_path (str): Path to the text file

            Returns:
                List[str]: List of document chunks
            """
            with open(file_path, 'r', encoding='utf-8') as file:
                # Read entire file
                content = file.read()

            # Split content into chunks (e.g., 10 lines per chunk)
            lines = content.split('\n')
            chunks = []
            chunk_size = 10  # Adjust based on your file structure

            for i in range(0, len(lines), chunk_size):
                chunk = '\n'.join(lines[i:i+chunk_size])
                chunks.append(chunk)

            return chunks

        def add_documents_from_directory(self, directory_path: str):
            """
            Add documents from all text files in a directory

            Args:
                directory_path (str): Path to directory containing text files
            """
            for filename in os.listdir(directory_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(directory_path, filename)
                    document_chunks = self.load_text_file(file_path)

                    for chunk in document_chunks:
                        # Embed document chunk
                        embedding = self.embedding_model.encode(chunk)

                        # Add to index and document store
                        self.index.add(np.array([embedding]))
                        self.documents.append(chunk)
                        self.document_embeddings.append(embedding)
                        self.document_metadata.append({
                            'filename': filename,
                            'chunk_index': len(self.document_metadata)
                        })

        def count_tokens(self, text: str) -> int:
            """
            Count tokens in a given text

            Args:
                text (str): Input text

            Returns:
                int: Number of tokens
            """
            return len(self.tokenizer.encode(text))

        def retrieve_and_truncate_context(self, query: str) -> str:
            """
            Retrieve relevant documents and truncate to fit token limit

            Args:
                query (str): Input query

            Returns:
                str: Concatenated context within token limit
            """
            # Retrieve relevant documents
            query_embedding = self.embedding_model.encode(query)
            distances, indices = self.index.search(
                np.array([query_embedding]),
                self.top_k_retrieval
            )

            # Collect and truncate context
            context = []
            current_tokens = 0

            for idx in indices[0]:
                doc = self.documents[idx]
                doc_tokens = self.count_tokens(doc)

                # Check if adding this document would exceed token limit
                if current_tokens + doc_tokens <= self.max_context_tokens:
                    context.append(doc)
                    current_tokens += doc_tokens
                else:
                    break

            return "\n\n".join(context)

        def generate_response(self, query: str) -> str:
            """
            Generate a response using retrieval-augmented generation

            Args:
                query (str): User's input query

            Returns:
                str: Generated response incorporating retrieved context
            """
            # Retrieve and truncate context
            retrieved_context = self.retrieve_and_truncate_context(query)

            # Construct context-aware prompt
            full_prompt = f"""
            Context Documents:
            {retrieved_context}

            Query: {query}

            Generate a comprehensive response that:
            1. Directly answers the query
            2. Incorporates relevant information from the context documents
            3. Provides clear and concise information
            """

            # Prepare API arguments
            api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
                input=full_prompt,
                model_kwargs=self.model_kwargs,
                model_type=ModelType.LLM
            )

            # Call API and parse response
            response = self.model_client.call(
                api_kwargs=api_kwargs,
                model_type=ModelType.LLM
            )
            response_text = self.model_client.parse_chat_completion(response)

            return response_text

The ``run_rag_pipeline`` function demonstrates how to use the ``AdalflowRAGPipeline``. It initializes the pipeline,
adds documents from a directory, and generates responses for a list of user queries. The function is generic
and can accommodate various LLM API clients, such as GroqAPIClient or OpenAIClient, highlighting the pipeline's
flexibility and modularity.


.. code-block:: python

    def run_rag_pipeline(model_client, model_kwargs, documents, queries):

        # Example usage of RAG pipeline
        rag_pipeline = AdalflowRAGPipeline(
            model_client=model_client,
            model_kwargs=model_kwargs,
            top_k_retrieval=1,  # Retrieve top 1 most relevant chunks
            max_context_tokens=800  # Limit context to 1500 tokens
        )

        # Add documents from a directory of text files
        rag_pipeline.add_documents_from_directory(documents)

        # Generate responses
        for query in queries:
            print(f"\nQuery: {query}")
            response = rag_pipeline.generate_response(query)
            print(f"Response: {response}")


This block provides an example of running the pipeline with different models and queries. It specifies:

- The document directory containing the text files.
- Example queries about topics such as the "Crystal Cavern" and "rare trees in Elmsworth."
- Configuration for Groq and OpenAI model parameters, including the model type, temperature, and token limits.

.. code-block:: python

    documents = '../../tutorials/assets/documents'

    queries = [
        "What year was the Crystal Cavern discovered?",
        "What is the name of the rare tree in Elmsworth?",
        "What local legend claim that Lunaflits surrounds?"
    ]

    groq_model_kwargs = {
        "model": "llama-3.2-1b-preview",  # Use 16k model for larger context
        "temperature": 0.1,
        "max_tokens": 800,
    }

    openai_model_kwargs = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.1,
        "max_tokens": 800,
    }
    # Below example shows that adalflow can be used in a genric manner for any api provider
    # without worrying about prompt and parsing results
    run_rag_pipeline(GroqAPIClient(), groq_model_kwargs, documents, queries)
    run_rag_pipeline(OpenAIClient(), openai_model_kwargs, documents, queries)

The example emphasizes that ``AdalflowRAGPipeline`` can interact seamlessly with multiple API providers,
enabling integration with diverse LLMs without modifying the core logic for prompt construction or
response parsing.


.. admonition:: API reference
   :class: highlight

   - :class:`utils.setup_env`
   - :class:`core.types.ModelType`
   - :class:`components.model_client.OpenAIClient`
   - :class:`components.model_client.GroqAPIClient`
