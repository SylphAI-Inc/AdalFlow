"""
Ollama Models Tutorial

This tutorial demonstrates how to use Ollama with AdalFlow to run local LLMs.
Ollama allows you to run open-source models locally without depending on external APIs.

Prerequisites:
1. Install Ollama from https://ollama.com/
2. Pull the model you want to use (e.g., `ollama pull qwen2:0.5b` or `ollama pull mistral`)
3. Ensure Ollama is running locally (default port: 11434)
"""

import asyncio
import requests
from typing import Optional

from adalflow.components.model_client import OllamaClient
from adalflow.core import Generator, Embedder


def check_ollama_connection(host: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(f"{host}/api/version", timeout=2)
        if response.status_code == 200:
            print("✅ Ollama is running!")
            print(f"Version: {response.json()}")
            return True
        else:
            print("❌ Ollama is not responding correctly")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Please make sure Ollama is running.")
        print("Run 'ollama serve' in your terminal to start Ollama.")
        return False


def list_available_models(host: str = "http://localhost:11434") -> None:
    """List all models available in your Ollama installation."""
    response = requests.get(f"{host}/api/tags")
    if response.status_code == 200:
        models = response.json().get('models', [])
        if models:
            print("Available models:")
            for model in models:
                size_gb = model.get('size', 0) / 1e9
                print(f"  - {model['name']} ({size_gb:.2f} GB)")
        else:
            print("No models found. Pull a model using: ollama pull qwen2:0.5b")
    else:
        print("Could not fetch models")


def basic_usage_example():
    """Basic text generation with Ollama."""
    print("\n=== Basic Usage Example ===\n")
    
    # Initialize the Generator with OllamaClient
    generator = Generator(
        model_client=OllamaClient(host="http://localhost:11434"),
        model_kwargs={"model": "qwen2:0.5b"}
    )
    
    # Generate a response
    response = generator({"input_str": "Hello, what can you do?"})
    print("Response:")
    print(response.data)
    
    return response


def text_generation_with_parameters():
    """Text generation with custom model parameters."""
    print("\n=== Text Generation with Parameters ===\n")
    
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
    
    response = generator({"input_str": "Explain quantum computing in simple terms"})
    print("Response:")
    print(response.data)
    
    return response


def synchronous_streaming_example():
    """Synchronous streaming for real-time output."""
    print("\n=== Synchronous Streaming Example ===\n")
    
    # Enable streaming
    stream_generator = Generator(
        model_client=OllamaClient(),
        model_kwargs={
            "model": "qwen2:0.5b",
            "stream": True
        }
    )
    
    output = stream_generator.call(
        prompt_kwargs={"input_str": "Tell me a short story about a robot"}
    )
    
    print("Streaming response:")
    # Access the raw streaming response
    for chunk in output.raw_response:
        if "message" in chunk:
            print(chunk["message"]["content"], end='', flush=True)
    print("\n")
    
    return output


async def asynchronous_streaming_example():
    """Asynchronous streaming for better performance."""
    print("\n=== Asynchronous Streaming Example ===\n")
    
    # Enable streaming
    stream_generator = Generator(
        model_client=OllamaClient(),
        model_kwargs={
            "model": "qwen2:0.5b",
            "stream": True
        }
    )
    
    # Using async streaming
    output = await stream_generator.acall(
        prompt_kwargs={"input_str": "What are the benefits of async programming?"}
    )
    
    print("Async streaming response:")
    # Access the raw async streaming response
    async for chunk in output.raw_response:
        if "message" in chunk:
            print(chunk["message"]["content"], end='', flush=True)
    print("\n")
    
    return output


async def async_non_streaming_example():
    """Asynchronous call without streaming."""
    print("\n=== Async Non-Streaming Example ===\n")
    
    generator = Generator(
        model_client=OllamaClient(),
        model_kwargs={"model": "qwen2:0.5b"}
    )
    
    # Async call
    output = await generator.acall(
        prompt_kwargs={"input_str": "What is machine learning?"}
    )
    
    print("Async response:")
    print(output.data)
    
    return output


def chat_vs_generate_api_example():
    """Demonstrate the difference between Chat and Generate APIs."""
    print("\n=== Chat vs Generate API Example ===\n")
    
    # Chat API (default) - uses conversation format
    chat_generator = Generator(
        model_client=OllamaClient(),
        model_kwargs={"model": "qwen2:0.5b"}
    )
    
    print("Chat API response:")
    chat_response = chat_generator({"input_str": "What is Python?"})
    print(chat_response.data[:200] + "...\n")
    
    # Generate API - uses raw prompt
    generate_generator = Generator(
        model_client=OllamaClient(),
        model_kwargs={
            "model": "qwen2:0.5b",
            "generate": True  # Use generate API instead of chat
        }
    )
    
    print("Generate API response:")
    gen_response = generate_generator({"input_str": "Python is"})
    print(gen_response.data[:200] + "...")
    
    return chat_response, gen_response


def text_embeddings_example(model: str = "nomic-embed-text"):
    """Generate text embeddings for semantic search."""
    print(f"\n=== Text Embeddings Example (model: {model}) ===\n")
    
    try:
        embedder = Embedder(
            model_client=OllamaClient(),
            model_kwargs={"model": model}
        )
        
        # Single text embedding
        text = "This is a sample text for embedding"
        embedding = embedder(input=text)
        
        if embedding.data and len(embedding.data) > 0:
            embedding_dim = len(embedding.data[0].embedding)
            print(f"✅ Successfully generated embedding")
            print(f"Embedding dimension: {embedding_dim}")
            print(f"First 10 values: {embedding.data[0].embedding[:10]}")
        else:
            print(f"❌ No embedding data returned")
            
        return embedding
        
    except Exception as e:
        print(f"❌ Error with {model}: {str(e)}")
        print(f"Model {model} may not be installed. Pull it with: ollama pull {model}")
        return None


def gpt_oss_reasoning_example():
    """Use GPT-OSS models for reasoning tasks."""
    print("\n=== GPT-OSS Reasoning Example ===\n")
    
    try:
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
            "input_str": "Solve this problem step by step: If a train travels 120 km in 2 hours, what is its average speed?"
        })
        
        # Access reasoning process if available
        if response.thinking:
            print("Thinking:")
            print(response.thinking)
            print("\n")
        
        print("Answer:")
        print(response.data)
        
        return response
        
    except Exception as e:
        print(f"❌ Error with gpt-oss:20b: {str(e)}")
        print("Model gpt-oss:20b may not be installed. Pull it with: ollama pull gpt-oss:20b")
        return None


def model_with_options_example():
    """Demonstrate all available model options."""
    print("\n=== Model with All Options Example ===\n")
    
    generator = Generator(
        model_client=OllamaClient(),
        model_kwargs={
            "model": "qwen2:0.5b",
            "options": {
                "seed": 42,              # Random seed for reproducibility
                "num_predict": 256,      # Maximum tokens to generate
                "temperature": 0.7,      # Creativity level (0.0-2.0)
                "top_k": 40,            # Number of top tokens to consider
                "top_p": 0.9,           # Cumulative probability cutoff
                "repeat_penalty": 1.1,   # Penalty for repeated tokens
                "num_ctx": 2048,        # Context window size
                "stop": ["\n", "###"],  # Stop sequences
                "mirostat": 0,          # Mirostat sampling (0=disabled)
                "mirostat_tau": 5.0,    # Balance between coherence and diversity
                "mirostat_eta": 0.1,    # Learning rate
            }
        }
    )
    
    response = generator({
        "input_str": "Write a haiku about programming"
    })
    
    print("Response with custom options:")
    print(response.data)
    
    return response


def main():
    """Run all examples."""
    print("=" * 60)
    print("Ollama Models Tutorial with AdalFlow")
    print("=" * 60)
    
    # Check Ollama connection
    if not check_ollama_connection():
        print("\n⚠️  Please start Ollama before running this tutorial.")
        print("Run: ollama serve")
        return
    
    # List available models
    print("\n")
    list_available_models()
    
    # Check for required models
    print("\n" + "=" * 60)
    print("Running Examples")
    print("=" * 60)
    
    print("\nNote: Some examples may fail if required models are not installed.")
    print("Install models with: ollama pull <model_name>")
    print("Recommended models: qwen2:0.5b, gpt-oss:20b, nomic-embed-text")
    
    try:
        # Run synchronous examples
        basic_usage_example()
        text_generation_with_parameters()
        synchronous_streaming_example()
        chat_vs_generate_api_example()
        model_with_options_example()
        
        # Try embeddings (may fail if model not installed)
        text_embeddings_example()
        
        # Try GPT-OSS (may fail if model not installed)
        gpt_oss_reasoning_example()
        
        # Run async examples
        print("\n" + "=" * 60)
        print("Running Async Examples")
        print("=" * 60)
        
        asyncio.run(async_examples())
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        print("Make sure Ollama is running and required models are installed.")


async def async_examples():
    """Run all async examples."""
    await async_non_streaming_example()
    await asynchronous_streaming_example()


if __name__ == "__main__":
    main()