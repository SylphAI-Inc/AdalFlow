"""Example script demonstrating Azure OpenAI client usage in AdalFlow."""

import os
import asyncio
from adalflow.components.model_client import AzureClient
from adalflow.core.generator import Generator
from adalflow.core.types import ModelType

# Demo configuration - Replace these with your actual values
DEMO_CONFIG = {
    "api_key": "your-api-key",  # From Azure Portal > Keys and Endpoint
    "azure_endpoint": "https://your-resource.openai.azure.com/",  # Your Azure OpenAI endpoint
    "api_version": "2024-02-15-preview",  # Current API version
    "deployment_name": "gpt-35-turbo"  # Your model deployment name
}

def setup_environment():
    """Setup environment variables if not already set."""
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("Setting up demo environment variables...")
        os.environ["AZURE_OPENAI_API_KEY"] = DEMO_CONFIG["api_key"]
        os.environ["AZURE_OPENAI_ENDPOINT"] = DEMO_CONFIG["azure_endpoint"]
        os.environ["AZURE_OPENAI_VERSION"] = DEMO_CONFIG["api_version"]
    else:
        print("Using existing environment variables...")

def test_chat_completion():
    """Test chat completion with Azure OpenAI."""
    print("\nTesting chat completion...")
    client = AzureClient()
    generator = Generator(
        model_client=client,
        model_kwargs={
            "model": DEMO_CONFIG["deployment_name"],
            "temperature": 0.7,
        },
        model_type=ModelType.LLM
    )

    # Single turn conversation
    response = generator("What is the capital of France?")
    print("\nChat Completion Response:")
    print(f"Content: {response.raw_response}")
    print(f"Usage: {response.usage}")
    print(f"Error: {response.error}")

    # Multi-turn conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What is its population?"}
    ]
    
    client = AzureClient(input_type="messages")
    generator = Generator(
        model_client=client,
        model_kwargs={
            "model": DEMO_CONFIG["deployment_name"],
            "temperature": 0.7,
        },
        model_type=ModelType.LLM
    )
    
    response = generator(messages)
    print("\nMulti-turn Conversation Response:")
    print(f"Content: {response.raw_response}")
    print(f"Usage: {response.usage}")
    print(f"Error: {response.error}")

def test_embeddings():
    """Test embeddings with Azure OpenAI."""
    print("\nTesting embeddings...")
    client = AzureClient()
    generator = Generator(
        model_client=client,
        model_kwargs={
            "model": "text-embedding-ada-002",  # Standard embedding model name
        },
        model_type=ModelType.EMBEDDER
    )

    # Single text embedding
    response = generator("Hello, world!")
    print("\nSingle Text Embedding Response:")
    print(f"Embedding shape: {len(response.raw_response)}")
    print(f"Usage: {response.usage}")
    print(f"Error: {response.error}")

    # Multiple text embeddings
    texts = ["Hello, world!", "How are you?", "Nice to meet you!"]
    response = generator(texts)
    print("\nMultiple Text Embeddings Response:")
    print(f"Number of embeddings: {len(response.raw_response)}")
    print(f"Usage: {response.usage}")
    print(f"Error: {response.error}")

async def test_async_chat():
    """Test async chat completion with Azure OpenAI."""
    print("\nTesting async chat completion...")
    client = AzureClient()
    generator = Generator(
        model_client=client,
        model_kwargs={
            "model": DEMO_CONFIG["deployment_name"],
            "temperature": 0.7,
        },
        model_type=ModelType.LLM
    )

    response = await generator.acall("What is the capital of France?")
    print("\nAsync Chat Completion Response:")
    print(f"Content: {response.raw_response}")
    print(f"Usage: {response.usage}")
    print(f"Error: {response.error}")

def test_streaming():
    """Test streaming chat completion with Azure OpenAI."""
    print("\nTesting streaming chat completion...")
    client = AzureClient()
    generator = Generator(
        model_client=client,
        model_kwargs={
            "model": DEMO_CONFIG["deployment_name"],
            "temperature": 0.7,
            "stream": True
        },
        model_type=ModelType.LLM
    )

    print("\nStreaming Chat Completion Response:")
    for chunk in generator("Tell me a short story about a cat."):
        if chunk.raw_response:
            print(chunk.raw_response, end="", flush=True)
    print("\n")

def check_requirements():
    """Check if all required packages are installed."""
    try:
        import openai
        import azure.identity
        import azure.mgmt.cognitiveservices
        print("All required packages are installed.")
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages:")
        print("poetry add openai azure-identity azure-mgmt-cognitiveservices")
        return False

if __name__ == "__main__":
    print("Azure OpenAI Client Test Script")
    print("==============================")
    
    if not check_requirements():
        exit(1)

    setup_environment()

    # Check for required environment variables
    required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_VERSION"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them before running this script.")
        print("\nYou can set them in your environment:")
        print("export AZURE_OPENAI_API_KEY='your-key'")
        print("export AZURE_OPENAI_ENDPOINT='your-endpoint'")
        print("export AZURE_OPENAI_VERSION='api-version'")
        print("\nOr update the DEMO_CONFIG in this script.")
        exit(1)

    print("\nStarting Azure OpenAI tests...")
    
    try:
        # Test synchronous operations
        test_chat_completion()
        test_embeddings()
        test_streaming()
        
        # Test asynchronous operations
        asyncio.run(test_async_chat())
        
        print("\nAll tests completed!")
    except Exception as e:
        print(f"\nError during testing: {e}")
        print("\nPlease check your Azure OpenAI setup and credentials.") 