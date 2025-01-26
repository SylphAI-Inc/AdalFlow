"""
OpenAI Vision and DALL-E Example with Error Testing

To test with different API keys:

1. First run with a valid key:
   export OPENAI_API_KEY='your_valid_key_here'
   python tutorials/vision_dalle_example.py

2. Then test with an invalid key:
   export OPENAI_API_KEY='abc123'
   python tutorials/vision_dalle_example.py

The script will show different GeneratorOutput responses based on the API key status.
"""

from adalflow.core import Generator
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.core.types import ModelType
import asyncio
import numpy as np
from dataclasses import dataclass
from typing import List
from numpy.linalg import norm

def test_basic_generation():
    """Test basic text generation"""
    client = OpenAIClient()  # For text/chat completion
    gen = Generator(
        model_client=client,
        model_kwargs={"model": "gpt-4o-mini", "max_tokens": 100},
        model_type=ModelType.LLM  # Explicitly specify model type
    )

    print("\n=== Testing Basic Generation ===")
    response = gen({"input_str": "Hello, world!"})
    print(f"Response: {response}")


def test_invalid_image_url():
    """Test Generator output with invalid image URL"""
    client = OpenAIClient()  # For vision tasks
    gen = Generator(
        model_client=client,
        model_kwargs={
            "model": "gpt-4o-mini",
            "images": "https://invalid.url/nonexistent.jpg",
            "max_tokens": 300,
        },
        model_type=ModelType.LLM  # Vision tasks use LLM type
    )

    print("\n=== Testing Invalid Image URL ===")
    response = gen({"input_str": "What do you see in this image?"})
    print(f"Response with invalid image URL: {response}")


def test_invalid_image_generation():
    """Test DALL-E generation with invalid parameters"""
    client = OpenAIClient()  # For image generation
    gen = Generator(
        model_client=client,
        model_kwargs={
            "model": "dall-e-3",
            "size": "invalid_size",  # Invalid size parameter
            "quality": "standard",
            "n": 1,
        },
        model_type=ModelType.IMAGE_GENERATION  # Specify image generation type
    )

    print("\n=== Testing Invalid DALL-E Parameters ===")
    response = gen({"input_str": "A cat"})
    print(f"Response with invalid DALL-E parameters: {response}")


def test_vision_and_generation():
    """Test both vision analysis and image generation"""
    # 1. Test Vision Analysis with LLM client
    vision_client = OpenAIClient()  # For vision tasks
    vision_gen = Generator(
        model_client=vision_client,
        model_kwargs={
            "model": "gpt-4o-mini",
            "images": "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
            "max_tokens": 300,
        },
        model_type=ModelType.LLM  # Vision tasks use LLM type
    )

    vision_response = vision_gen(
        {"input_str": "What do you see in this image? Be detailed but concise."}
    )
    print("\n=== Vision Analysis ===")
    print(f"Description: {vision_response.raw_response}")

    # 2. Test DALL-E Image Generation
    dalle_client = OpenAIClient()  # For image generation
    dalle_gen = Generator(
        model_client=dalle_client,
        model_kwargs={
            "model": "dall-e-3",
            "size": "1024x1024",
            "quality": "standard",
            "n": 1,
        },
        model_type=ModelType.IMAGE_GENERATION  # Specify image generation type
    )

    # For image generation, input_str becomes the prompt
    response = dalle_gen(
        {"input_str": "A happy siamese cat playing with a red ball of yarn"}
    )
    print("\n=== DALL-E Generation ===")
    print(f"Generated Image URL: {response.data}")


if __name__ == "__main__":
    print("Starting OpenAI Vision and DALL-E test...\n")

    # Run all tests - they will show errors if API key is invalid/empty
    test_basic_generation()
    test_invalid_image_url()
    test_invalid_image_generation()
    test_vision_and_generation()
