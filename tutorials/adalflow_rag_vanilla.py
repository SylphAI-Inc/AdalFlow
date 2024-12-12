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
        top_k_retrieval=1,
    ):
        """
        Initialize RAG Pipeline with embedding and retrieval components

        Args:
            embedding_model (str): Sentence transformer model for embeddings
            vector_dim (int): Dimension of embedding vectors
            top_k_retrieval (int): Number of documents to retrieve
        """
        # Initialize model client for generation
        self.model_client = model_client

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize FAISS index for vector similarity search
        self.index = IndexFlatL2(vector_dim)

        # Store document texts and their embeddings
        self.documents = []
        self.document_embeddings = []

        # Retrieval parameters
        self.top_k_retrieval = top_k_retrieval

        # Conversation history and context
        self.conversation_history = ""
        self.model_kwargs = model_kwargs

    def add_documents(self, documents: List[str]):
        """
        Add documents to the RAG pipeline's knowledge base

        Args:
            documents (List[str]): List of document texts to add
        """
        for doc in documents:
            # Embed document
            embedding = self.embedding_model.encode(doc)

            # Add to index and document store
            self.index.add(np.array([embedding]))
            self.documents.append(doc)
            self.document_embeddings.append(embedding)

    def retrieve_relevant_docs(self, query: str) -> List[str]:
        """
        Retrieve most relevant documents for a given query

        Args:
            query (str): Input query to find relevant documents

        Returns:
            List[str]: Top k most relevant documents
        """
        # Embed query
        query_embedding = self.embedding_model.encode(query)

        # Perform similarity search
        distances, indices = self.index.search(
            np.array([query_embedding]), self.top_k_retrieval
        )

        # Retrieve and return top documents
        return [self.documents[i] for i in indices[0]]

    def generate_response(self, query: str) -> str:
        """
        Generate a response using retrieval-augmented generation

        Args:
            query (str): User's input query

        Returns:
            str: Generated response incorporating retrieved context
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_relevant_docs(query)

        # Construct context-aware prompt
        context = "\n\n".join([f"Context Document: {doc}" for doc in retrieved_docs])
        full_prompt = f"""
        Context:
        {context}

        Query: {query}

        Generate a comprehensive and informative response that:
        1. Uses the provided context documents
        2. Directly answers the query
        3. Incorporates relevant information from the context
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

        # Update conversation history
        self.conversation_history += f"\nQuery: {query}\nResponse: {response_text}"

        return response_text


def run_rag_pipeline(model_client, model_kwargs, documents, queries):
    rag_pipeline = AdalflowRAGPipeline(
        model_client=model_client, model_kwargs=model_kwargs
    )

    rag_pipeline.add_documents(documents)

    # Generate responses
    for query in queries:
        print(f"\nQuery: {query}")
        response = rag_pipeline.generate_response(query)
        print(f"Response: {response}")


def main():
    setup_env()

    # ajithvcoder's statements are added so that we can validate that the LLM is generating from these lines only
    documents = [
        "ajithvcoder is a good person whom the world knows as Ajith Kumar, ajithvcoder is his nick name that AjithKumar gave himself",
        "The Eiffel Tower is a famous landmark in Paris, built in 1889 for the World's Fair.",
        "ajithvcoder likes Hyderabadi panner dum briyani much.",
        "The Louvre Museum in Paris is the world's largest art museum, housing thousands of works of art.",
        "ajithvcoder has a engineering degree and he graduated on May, 2016.",
    ]

    # Questions related to ajithvcoder's are added so that we can validate
    # that the LLM is generating from above given lines only
    queries = [
        "Does Ajith Kumar has any nick name ?",
        "What is the ajithvcoder's favourite food?",
        "When did ajithvcoder graduated ?",
    ]

    groq_model_kwargs = {
        "model": "llama-3.2-1b-preview",  # Use 16k model for larger context
        "temperature": 0.1,
        "max_tokens": 800,
    }

    openai_model_kwargs = {
        "model": "gpt-3.5-turbo",  # Use 16k model for larger context
        "temperature": 0.1,
        "max_tokens": 800,
    }

    # Below example shows that adalflow can be used in a genric manner for any api provider
    # without worrying about prompt and parsing results
    run_rag_pipeline(GroqAPIClient(), groq_model_kwargs, documents, queries)
    run_rag_pipeline(OpenAIClient(), openai_model_kwargs, documents, queries)


if __name__ == "__main__":
    main()
