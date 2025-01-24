"""
Azure AI Client Example for AdalFlow
==================================

This example demonstrates how to use the AdalFlow Azure AI client with both API key
and Azure AD authentication methods. It shows various features including chat completions,
streaming responses, and text embeddings.

Setup Instructions:
-----------------
1. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

2. Configure environment variables:
   Create a .env file with the following variables:
   ```
   # Azure OpenAI API Configuration
   AZURE_OPENAI_API_KEY="2r27KgyhYBPHA2h1tR51BUZegp64jNRm40QW7O0xyqQRu1q6EGGsJQQJ99BAACfhMk5XJ3w3AAAAACOGuKK6"
   AZURE_OPENAI_ENDPOINT="https://filip-m67wzyeo-swedencentral.cognitiveservices.azure.com/"
   AZURE_OPENAI_VERSION="2024-02-15-preview"

   # Model Deployments
   AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini"
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-ada-002-2"
   ```

3. Run the example:
   ```bash
   poetry run python examples/azure_client_example.py
   ```

Features Demonstrated:
--------------------
1. Multiple Authentication Methods:
   - API key-based authentication
   - Azure AD authentication using DefaultAzureCredential

2. Chat Completions:
   - Regular chat completion
   - Streaming chat completion
   - System and user message handling

3. Text Embeddings:
   - Generate embeddings for multiple texts
   - Embedding dimension output
"""

import os
import asyncio
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential

from adalflow.components.model_client.azureai_client import AzureAIClient
from adalflow.core import Generator, Embedder
from adalflow.core.types import ModelType
from adalflow.utils import setup_env, get_logger

# Setup logging
log = get_logger(level="DEBUG")


def init_azure_client(use_aad: bool = False) -> AzureAIClient:
    """Initialize Azure client with either API key or AAD authentication.

    Args:
        use_aad (bool): If True, uses Azure AD authentication. If False, uses API key.

    Returns:
        AzureAIClient: Configured client instance
    """
    if use_aad:
        # Using Azure AD authentication
        credential = DefaultAzureCredential()
        return AzureAIClient(
            api_version=os.getenv("AZURE_OPENAI_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            credential=credential,
        )
    else:
        # Using API key authentication
        return AzureAIClient(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )


async def test_chat_completion(client: AzureAIClient, stream: bool = False):
    """Test chat completion with optional streaming."""
    print("\n=== Testing Chat Completion ===")

    # Set model type for chat completion
    client.model_type = ModelType.LLM

    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    print(f"Using deployment: {deployment_name}")

    # Initialize Generator with the Azure client
    generator = Generator(
        model_client=client,
        model_kwargs={
            "model": deployment_name,
            "stream": stream,
            "temperature": 0.7,
            "max_tokens": 500,
        },
    )

    # Example system and user prompts
    system_prompt = "You are a helpful assistant."
    user_prompt = "What are the top 3 tourist attractions in Paris?"

    # Format the input for the generator
    input_text = f"<START_OF_SYSTEM_PROMPT>{system_prompt}<END_OF_SYSTEM_PROMPT><START_OF_USER_PROMPT>{user_prompt}<END_OF_USER_PROMPT>"

    try:
        # Generate response using acall
        response = await generator.acall(prompt_kwargs={"input_str": input_text})

        if response.error:
            print(f"Error: {response.error}")
            return

        if stream and response.data:
            async for chunk in response.data:
                print(chunk, end="", flush=True)
            print("\n")
        else:
            print(response.raw_response)
    except Exception as e:
        print(f"Error during chat completion: {str(e)}")


async def test_embeddings(client: AzureAIClient):
    """Test text embeddings functionality."""
    print("\n=== Testing Embeddings ===")

    # Set model type for embeddings
    client.model_type = ModelType.EMBEDDER

    deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    print(f"Using embedding deployment: {deployment_name}")

    # Initialize Embedder with the Azure client
    embedder = Embedder(
        model_client=client,
        model_kwargs={
            "model": deployment_name,
        },
    )

    # Example texts to embed
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Paris is the capital of France",
    ]

    try:
        # Generate embeddings
        embeddings = await embedder.acall(input=texts)
        if embeddings.error:
            print(f"Error: {embeddings.error}")
            return

        print(f"Generated {len(embeddings.data)} embeddings")
        print(f"Embedding dimension: {len(embeddings.data[0].embedding)}")
    except Exception as e:
        print(f"Error during embedding generation: {str(e)}")


async def main():
    """Main function demonstrating all features."""
    # Load environment variables
    load_dotenv()
    setup_env()

    # Print available deployments
    print("\nEnvironment Configuration:")
    print(f"Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"Chat Model Deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT')}")
    print(
        f"Embedding Model Deployment: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')}"
    )

    # Test with API key authentication
    print("\nTesting with API key authentication:")
    client = init_azure_client(use_aad=False)

    try:
        # Test regular chat completion (no streaming)
        print("\nTesting Chat Completion:")
        await test_chat_completion(client, stream=False)

        # Test embeddings
        print("\nTesting Embeddings:")
        await test_embeddings(client)

        print("\nAll tests completed successfully!")

    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        if "429" in str(e):
            print(
                "Hit rate limit - the basic functionality works, but we're being throttled."
            )
        elif "DeploymentNotFound" in str(e):
            print(
                "Check if your deployment names in .env match exactly with what's in Azure Portal."
            )


if __name__ == "__main__":
    asyncio.run(main())
