import os
import tiktoken
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2

from adalflow.components.model_client import GroqAPIClient, OpenAIClient
from adalflow.core.types import ModelType
from adalflow.utils import setup_env

"""
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers==3.3.1
pip install faiss-cpu==1.9.0.post1
"""


class AdalflowRAGPipeline:
    def __init__(
        self,
        model_client=None,
        model_kwargs=None,
        embedding_model="all-MiniLM-L6-v2",
        vector_dim=384,
        top_k_retrieval=3,
        max_context_tokens=800,
    ):
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
        with open(file_path, "r", encoding="utf-8") as file:
            # Read entire file
            content = file.read()

        # Split content into chunks (e.g., 10 lines per chunk)
        lines = content.split("\n")
        chunks = []
        chunk_size = 10  # Adjust based on your file structure

        for i in range(0, len(lines), chunk_size):
            chunk = "\n".join(lines[i : i + chunk_size])
            chunks.append(chunk)

        return chunks

    def add_documents_from_directory(self, directory_path: str):
        """
        Add documents from all text files in a directory

        Args:
            directory_path (str): Path to directory containing text files
        """
        for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory_path, filename)
                document_chunks = self.load_text_file(file_path)

                for chunk in document_chunks:
                    # Embed document chunk
                    embedding = self.embedding_model.encode(chunk)

                    # Add to index and document store
                    self.index.add(np.array([embedding]))
                    self.documents.append(chunk)
                    self.document_embeddings.append(embedding)
                    self.document_metadata.append(
                        {
                            "filename": filename,
                            "chunk_index": len(self.document_metadata),
                        }
                    )

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
            np.array([query_embedding]), self.top_k_retrieval
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
            input=full_prompt, model_kwargs=self.model_kwargs, model_type=ModelType.LLM
        )

        # Call API and parse response
        response = self.model_client.call(
            api_kwargs=api_kwargs, model_type=ModelType.LLM
        )
        response_text = self.model_client.parse_chat_completion(response)

        return response_text


def run_rag_pipeline(model_client, model_kwargs, documents, queries):

    # Example usage of RAG pipeline
    rag_pipeline = AdalflowRAGPipeline(
        model_client=model_client,
        model_kwargs=model_kwargs,
        top_k_retrieval=2,  # Retrieve top 3 most relevant chunks
        max_context_tokens=800,  # Limit context to 1500 tokens
    )

    # Add documents from a directory of text files
    rag_pipeline.add_documents_from_directory(documents)

    # Generate responses
    for query in queries:
        print(f"\nQuery: {query}")
        response = rag_pipeline.generate_response(query)
        print(f"Response: {response}")


def main():
    setup_env()

    documents = "./tutorials/assets/documents"

    queries = [
        "What year was the Crystal Cavern discovered?",
        "What is the name of the rare tree in Elmsworth?",
        "What local legend claim that Lunaflits surrounds?",
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


if __name__ == "__main__":
    main()
