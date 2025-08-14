"""Integration tests. Requires OpenAI API key and internet access."""

# This doc shows how to use all different OpenAI models and features in the Generator class.
# Demonstrates text generation, multimodal (vision), reasoning, and image generation capabilities.

import adalflow as adal
from adalflow.utils.logger import get_logger
import base64
from pathlib import Path

# log = get_logger(enable_file=False, level="DEBUG")

from adalflow.utils import setup_env
setup_env()  # Ensure environment variables are set, especially OPENAI_API_KEY


def test_basic_text_generation():
    """Basic text generation with GPT models."""
    print("\n=== Basic Text Generation ===")
    
    openai_llm = adal.Generator(
        model_client=adal.OpenAIClient(),
        model_kwargs={"model": "gpt-4o-mini"},
    )

    prompt_kwargs = {"input_str": "What is the meaning of life in one sentence?"}
    response = openai_llm(prompt_kwargs)

    print(f"Response: {response.data}")
    print(f"Usage: {response.usage}")


def test_streaming_response():
    """Streaming responses for real-time output."""
    print("\n=== Streaming Response ===")
    
    # Import the helper function for extracting text from streaming events
    from adalflow.components.model_client.utils import extract_text_from_response_stream
    
    openai_llm = adal.Generator(
        model_client=adal.OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o-mini",
            "stream": True
        },
    )

    prompt_kwargs = {"input_str": "Tell me a short story about a robot."}
    response = openai_llm(prompt_kwargs)
    
    print("Streaming output: ")
    
    # When streaming is enabled, raw_response contains the stream of events
    # Use the helper function to extract text from the Response API events
    if hasattr(response.raw_response, '__iter__'):
        for event in response.raw_response:
            # Use the helper to extract text from ResponseTextDeltaEvent
            text = extract_text_from_response_stream(event)
            if text:
                print(text, end="", flush=True)
    else:
        # Non-streaming fallback
        print(response.data)
    
    print()  # New line at the end


def test_multimodal_with_url():
    """Vision model with image from URL."""
    print("\n=== Multimodal with URL Image ===")
    
    openai_llm = adal.Generator(
        model_client=adal.OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o",
            "images": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        },
    )

    prompt_kwargs = {
        "input_str": "Describe this image in detail. What's the mood and atmosphere?"
    }
    response = openai_llm(prompt_kwargs)

    print(f"Image analysis: {response.data}")


def test_multimodal_with_local_image():
    """Vision model with local image file."""
    print("\n=== Multimodal with Local Image ===")
    
    import urllib.request
    import os
    
    # Download a test image from URL and save it locally
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    local_image_path = "temp_test_image.jpg"
    
    try:
        # Download the image
        print(f"Downloading test image to {local_image_path}...")
        urllib.request.urlretrieve(image_url, local_image_path)
        
        # Verify the file was created
        if os.path.exists(local_image_path):
            print(f"Image saved successfully ({os.path.getsize(local_image_path)} bytes)")
        
        # Use the local image with OpenAI
        openai_llm = adal.Generator(
            model_client=adal.OpenAIClient(),
            model_kwargs={
                "model": "gpt-4o",
                "images": local_image_path  # Pass the local file path
            },
        )

        prompt_kwargs = {
            "input_str": "Describe this nature scene. What do you see in the image?"
        }
        
        response = openai_llm(prompt_kwargs)
        print(f"Image analysis: {response.data}")
        
    except Exception as e:
        print(f"Error processing image: {e}")
    
    finally:
        # Clean up - delete the temporary image file
        if os.path.exists(local_image_path):
            os.remove(local_image_path)
            print(f"Cleaned up temporary file: {local_image_path}")


def test_multimodal_with_base64():
    """Vision model with pre-encoded base64 image."""
    print("\n=== Multimodal with Base64 Image ===")
    
    # Example of manually encoding an image (in practice, you'd have a real image)
    # base64_image = base64.b64encode(open("image.jpg", "rb").read()).decode('utf-8')
    
    # For demo, using a tiny 1x1 red pixel PNG
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    openai_llm = adal.Generator(
        model_client=adal.OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o",
            "images": f"data:image/png;base64,{base64_image}"  # Data URI format
        },
    )

    prompt_kwargs = {
        "input_str": "What do you see in this image?"
    }
    response = openai_llm(prompt_kwargs)

    print(f"Base64 image analysis: {response.data}")


def test_multiple_images():
    """Vision model with multiple images for comparison."""
    print("\n=== Multiple Images Comparison ===")
    
    openai_llm = adal.Generator(
        model_client=adal.OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o",
            "images": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
            ]
        },
    )

    prompt_kwargs = {
        "input_str": "Compare these two images. What are the main differences?"
    }
    response = openai_llm(prompt_kwargs)

    print(f"Comparison result: {response.data}")


def test_reasoning_model_5():
    """O1 reasoning model for complex problem solving."""
    print("\n=== gpt-5 Reasoning Model ===")
    
    # Note: O1 models require access and may have different pricing
    openai_llm = adal.Generator(
        model_client=adal.OpenAIClient(),
        # model_type=adal.ModelType.LLM_REASONING,
        model_kwargs={
            "model": "gpt-5",  # or "o1-mini",
            "reasoning": {
                "effort": "medium",  # low, medium, high
                "summary": "auto"    # detailed, auto, none
            }
        },
    )

    prompt_kwargs = {
        "input_str": "Solve this step by step: If a train travels 120 miles in 2 hours, and then 180 miles in 3 hours, what is its average speed for the entire journey?"
    }
    
    try:
        response = openai_llm(prompt_kwargs)
        print(f"Solution: {response.data}")
        if response.thinking:
            print(f"Reasoning process: {response.thinking}...")  # Show first 200 chars
    except Exception as e:
        print(f"gpt-5 model not available: {e}")
        print("gpt-5 models require special access")


def test_image_generation_tools():
    """Image generation using the new tools API."""
    print("\n=== Image Generation via Tools (New API) ===")
    
    openai_llm = adal.Generator(
        model_client=adal.OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o-mini",  # Any model that supports tools
            "tools": [{"type": "image_generation"}]
        },
    )

    prompt_kwargs = {
        "input_str": "Generate an image of a futuristic city with flying cars at sunset, cyberpunk style"
    }
    
    try:
        response = openai_llm(prompt_kwargs)

        print("Image generation response:")
        
        
        # With tools API, text goes to data, images go to images field
        if response.data:
            print(f"Text response: {response.data}")
        
        if response.images:
            print("Image(s) generated!")
            
            # Use the convenient save_images helper
            saved_paths = response.save_images(
                directory="output",
                prefix="futuristic_city",
                format="png"
            )
            print(f"Images saved to: {saved_paths}")
            
            # Or access raw base64 data directly
            if isinstance(response.images, str):
                print(f"Single image (base64 length: {len(response.images)})")
            else:
                print(f"Multiple images: {len(response.images)}")
                
    except Exception as e:
        print(f"Image generation via tools failed: {e}")


def test_mixed_text_and_image_generation():
    """Generate both text and images in a single call."""
    print("\n=== Mixed Text and Image Generation ===")
    
    openai_llm = adal.Generator(
        model_client=adal.OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o",
            "tools": [{"type": "image_generation"}]
        },
    )

    prompt_kwargs = {
        "input_str": "Write a haiku about mountains and then generate an image that captures the essence of the haiku"
    }
    
    try:
        response = openai_llm(prompt_kwargs)
        
        # Access text content
        if response.data:
            print(f"Haiku:\n{response.data}\n")
        
        # Access generated images
        if response.images:
            print(f"Generated {len(response.images)} image(s)")
            # Save with custom format
            paths = response.save_images(
                prefix="haiku_mountain",
                format="jpg",
            )
            print(f"Image saved: {paths}")
            
    except Exception as e:
        print(f"Mixed generation failed: {e}")


def test_embeddings():
    """Generate text embeddings for similarity search."""
    print("\n=== Text Embeddings ===")
    
    from adalflow.core import Embedder
    
    embedder = Embedder(
        model_client=adal.OpenAIClient(),
        model_kwargs={"model": "text-embedding-3-small"}
    )
    
    texts = [
        "The weather is beautiful today",
        "It's a sunny and pleasant day",
        "I love programming in Python"
    ]
    
    embeddings = embedder(input=texts)
    
    print(f"Generated {len(embeddings.data)} embeddings")
    if embeddings.data:
        print(f"Embedding dimension: {len(embeddings.data[0].embedding)}")
        print(f"First embedding preview: {embeddings.data[0].embedding[:5]}...")


def test_custom_api_endpoint():
    """Use OpenAI-compatible APIs from other providers."""
    print("\n=== Custom API Endpoint ===")
    
    # Example with a custom provider (e.g., local LLM, Azure, etc.)
    custom_client = adal.OpenAIClient(
        base_url="https://api.custom-provider.com/v1/",  # Replace with actual endpoint
        api_key="your-api-key",
        headers={"X-Custom-Header": "value"}
    )
    
    openai_llm = adal.Generator(
        model_client=custom_client,
        model_kwargs={"model": "custom-model-name"},
    )
    
    prompt_kwargs = {"input_str": "Hello, how are you?"}
    
    try:
        response = openai_llm(prompt_kwargs)
        print(f"Custom API response: {response.data}")
    except Exception as e:
        print(f"Custom API not configured: {e}")
        print("Replace with your actual API endpoint and credentials")




if __name__ == "__main__":
    # Setup environment (reads from .env file or uses environment variables)
    adal.setup_env()
    
    # Ensure output directory exists
    Path("output").mkdir(exist_ok=True)
    
    print("=" * 60)
    print("OpenAI Models Tutorial - Comprehensive Feature Showcase")
    print("=" * 60)
    
    # Basic features
    test_basic_text_generation()
    test_streaming_response()
    
    # # Multimodal (Vision)
    # test_multimodal_with_url()
    # test_multimodal_with_local_image()
    # test_multimodal_with_base64()
    # test_multiple_images()
    
    # # Reasoning models
    # test_reasoning_model_5()
    
    # # Image generation
    # test_image_generation_legacy()
    # test_image_generation_tools()
    # test_mixed_text_and_image_generation()
    
    # # Other features
    # test_embeddings()
    # test_custom_api_endpoint()
    
    # print("\n" + "=" * 60)
    # print("Tutorial completed! Check the output directory for generated images.")
    # print("=" * 60)