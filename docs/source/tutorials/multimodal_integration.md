# Multimodal Integration in AdalFlow

## Overview

AdalFlow now supports multimodal inputs (text + images) for OpenAI's vision-capable models through the `responses.create` API. This integration allows you to:

1. **Process images with GPT-4o and GPT-4.1 vision models**
2. **Use reasoning models (O1, O3) with image inputs**
3. **Handle both local images and URLs**
4. **Mix multiple image sources in a single request**

## API Format

The OpenAI `responses.create` API expects images in the following format:

```python
response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "what's in this image?"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        }
    ],
)
```

## Implementation Details

### 1. Image Encoding Function

Added to `adalflow/core/functional.py`:

```python
def encode_image(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
```

### 2. OpenAIClient Updates

The `convert_inputs_to_api_kwargs` method in `OpenAIClient` now:

- Detects when `images` are provided in `model_kwargs`
- Automatically formats images for the `responses.create` API
- Handles both URLs and local files
- Encodes local images to base64 automatically

### 3. Usage in Generator

The Generator can now accept images through `model_kwargs`:

```python
gen = Generator(
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o"},
    model_type=ModelType.LLM
)

response = gen(
    prompt_kwargs={"input_str": "What's in this image?"},
    model_kwargs={
        "images": "path/to/image.jpg"  # or URL
    }
)
```

## Usage Examples

### Basic Image Analysis

```python
from adalflow import Generator
from adalflow.components.model_client import OpenAIClient
from adalflow.core.types import ModelType

client = OpenAIClient()
gen = Generator(
    model_client=client,
    model_kwargs={"model": "gpt-4o"},
    model_type=ModelType.LLM
)

# With URL
response = gen(
    prompt_kwargs={"input_str": "Describe this image"},
    model_kwargs={
        "images": "https://example.com/image.jpg"
    }
)

# With local file
response = gen(
    prompt_kwargs={"input_str": "What objects are in this image?"},
    model_kwargs={
        "images": "/path/to/local/image.png"
    }
)
```

### Multiple Images

```python
# Compare multiple images
response = gen(
    prompt_kwargs={"input_str": "Compare these images"},
    model_kwargs={
        "images": [
            "https://example.com/image1.jpg",
            "/path/to/local/image2.png"
        ]
    }
)
```

### With Reasoning Models

```python
gen = Generator(
    model_client=client,
    model_kwargs={
        "model": "o3",
        "reasoning_effort": "high"
    },
    model_type=ModelType.LLM_REASONING
)

response = gen(
    prompt_kwargs={"input_str": "Analyze the patterns in this image"},
    model_kwargs={
        "images": "complex_chart.png"
    }
)
```

### Manual Image Encoding

If you need more control over image encoding:

```python
from adalflow.core.functional import encode_image

base64_img = encode_image("/path/to/image.jpg")

response = gen(
    prompt_kwargs={"input_str": "What's in this image?"},
    model_kwargs={
        "images": {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{base64_img}"
        }
    }
)
```

## Supported Models

### Vision Models
- **GPT-4o**: Multimodal model with vision capabilities
- **GPT-4.1**: Latest vision model
- **GPT-4-turbo-vision**: Vision-enabled turbo model

### Reasoning Models with Vision
- **O1**: Basic reasoning with image analysis
- **O3**: Advanced reasoning with image understanding
- **O3-mini**: Lightweight reasoning model with vision

## Best Practices

1. **Image Size**: Keep images reasonably sized (< 20MB)
2. **Format**: JPEG, PNG, GIF, and WebP are supported
3. **URLs vs Local**: 
   - Use URLs for publicly accessible images
   - Use local files for private/sensitive images
4. **Detail Level**: Set `detail: "high"` for detailed analysis
5. **Multiple Images**: Limit to 2-3 images per request for best performance

## Error Handling

The integration includes proper error handling for:
- Missing image files
- Permission errors
- Invalid image formats
- Network errors for URLs

```python
try:
    response = gen(
        prompt_kwargs={"input_str": "Analyze this"},
        model_kwargs={"images": "nonexistent.jpg"}
    )
except FileNotFoundError as e:
    print(f"Image not found: {e}")
except Exception as e:
    print(f"Error: {e}")
```

## Migration from Chat Completions API

If you were using the older chat completions API format:

**Old format:**
```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }
]
```

**New format (handled automatically):**
```python
# Just pass images in model_kwargs
response = gen(
    prompt_kwargs={"input_str": "What's in this image?"},
    model_kwargs={"images": image_url}
)
```

## Future Enhancements

1. **Automatic model selection** based on image presence
2. **Image preprocessing** (resize, compress)
3. **Batch image processing** optimization
4. **Support for video frames**
5. **Integration with other multimodal models**

## Troubleshooting

### Common Issues

1. **"Model does not support images"**
   - Ensure you're using a vision-capable model (gpt-4o, gpt-4.1, etc.)

2. **"Image file not found"**
   - Check the file path is absolute or relative to the working directory

3. **"Invalid image format"**
   - Ensure the image is in a supported format (JPEG, PNG, GIF, WebP)

4. **Rate limiting with large images**
   - Consider resizing images or using lower detail setting

## API Reference

### `encode_image(image_path: str) -> str`
Encode an image file to base64 string.

**Parameters:**
- `image_path`: Path to the image file

**Returns:**
- Base64 encoded string of the image

**Raises:**
- `FileNotFoundError`: If image file doesn't exist
- `PermissionError`: If no permission to read file
- `Exception`: For other encoding errors

### Generator `model_kwargs` for images

**Parameters:**
- `images`: Can be:
  - String: Single image path or URL
  - List: Multiple image paths/URLs
  - Dict: Pre-formatted image object
- `detail`: (Optional) "auto", "low", or "high"

## Conclusion

The multimodal integration in AdalFlow provides a seamless way to work with vision-capable models through the OpenAI responses.create API. The implementation handles the complexity of image encoding and formatting, allowing developers to focus on their applications.