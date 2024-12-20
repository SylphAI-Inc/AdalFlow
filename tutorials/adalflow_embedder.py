import os
from getpass import getpass
from adalflow.core.embedder import Embedder, BatchEmbedder
from adalflow.components.model_client import OpenAIClient, TransformersClient
from adalflow.core.types import Embedding, EmbedderOutput
from adalflow.core.functional import normalize_vector
from typing import List
from adalflow.core.component import Component
from copy import deepcopy


class DecreaseEmbeddingDim(Component):
    def __init__(self, old_dim: int, new_dim: int, normalize: bool = True):
        super().__init__()
        self.old_dim = old_dim
        self.new_dim = new_dim
        self.normalize = normalize
        assert self.new_dim < self.old_dim, "new_dim should be less than old_dim"

    def call(self, input: List[Embedding]) -> List[Embedding]:
        output: EmbedderOutput = deepcopy(input)
        for embedding in output.data:
            old_embedding = embedding.embedding
            new_embedding = old_embedding[: self.new_dim]
            if self.normalize:
                new_embedding = normalize_vector(new_embedding)
            embedding.embedding = new_embedding
        return output.data

    def _extra_repr(self) -> str:
        repr_str = f"old_dim={self.old_dim}, new_dim={self.new_dim}, normalize={self.normalize}"
        return repr_str


def setup_api_keys():
    # Prompt user to enter their API keys securely
    openai_api_key = getpass("Please enter your OpenAI API key: ")
    groq_api_key = getpass("Please enter your GROQ API key: ")

    # Set environment variables
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["GROQ_API_KEY"] = groq_api_key
    print("API keys have been set.")


def test_openai_embedder():
    print("\nTesting OpenAI Embedder:")
    model_kwargs = {
        "model": "text-embedding-3-small",
        "dimensions": 256,
        "encoding_format": "float",
    }

    query = "What is the capital of China?"
    queries = [query] * 100

    embedder = Embedder(model_client=OpenAIClient(), model_kwargs=model_kwargs)

    # Test single query
    output = embedder(query)
    print(
        f"Single query - Length: {output.length}, Dimension: {output.embedding_dim}, Normalized: {output.is_normalized}"
    )

    # Test batch queries
    output = embedder(queries)
    print(f"Batch queries - Length: {output.length}, Dimension: {output.embedding_dim}")


def test_local_embedder():
    print("\nTesting Local Embedder (HuggingFace):")
    model_kwargs = {"model": "thenlper/gte-base"}
    local_embedder = Embedder(
        model_client=TransformersClient(), model_kwargs=model_kwargs
    )

    query = "What is the capital of China?"
    queries = [query] * 100

    # Test single query
    output = local_embedder(query)
    print(
        f"Single query - Length: {output.length}, Dimension: {output.embedding_dim}, Normalized: {output.is_normalized}"
    )

    # Test batch queries
    output = local_embedder(queries)
    print(
        f"Batch queries - Length: {output.length}, Dimension: {output.embedding_dim}, Normalized: {output.is_normalized}"
    )


def test_custom_embedder():
    print("\nTesting Custom Embedder with Dimension Reduction:")
    model_kwargs = {"model": "thenlper/gte-base"}
    local_embedder_256 = Embedder(
        model_client=TransformersClient(),
        model_kwargs=model_kwargs,
        output_processors=DecreaseEmbeddingDim(768, 256),
    )

    query = "What is the capital of China?"
    output = local_embedder_256(query)
    print(
        f"Reduced dimension output - Length: {output.length}, Dimension: {output.embedding_dim}, Normalized: {output.is_normalized}"
    )


def test_batch_embedder():
    print("\nTesting Batch Embedder:")
    model_kwargs = {"model": "thenlper/gte-base"}
    local_embedder = Embedder(
        model_client=TransformersClient(), model_kwargs=model_kwargs
    )
    batch_embedder = BatchEmbedder(embedder=local_embedder, batch_size=100)

    query = "What is the capital of China?"
    queries = [query] * 1000

    print("Starting batch processing...")
    response = batch_embedder(queries)
    print(f"Batch processing complete - Total queries processed: {len(queries)}")
    print(f"Response - Length: {response.length}, Dimension: {response.embedding_dim}")


def main():
    # Setup API keys
    setup_api_keys()

    # Run all tests
    test_openai_embedder()
    test_local_embedder()
    test_custom_embedder()
    test_batch_embedder()


if __name__ == "__main__":
    main()
