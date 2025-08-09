"Helpers for model client for integrating models and parsing the output."

from typing import Union, List, Dict, Any, Optional
import os
from adalflow.core.types import EmbedderOutput, Embedding, Usage
from adalflow.core.functional import encode_image


def parse_embedding_response(
    api_response,
) -> EmbedderOutput:
    r"""Parse embedding model output from the API response to EmbedderOutput.

    Follows the OpenAI API response pattern.
    """
    # Assuming `api_response` has `.embeddings` and `.usage` attributes
    # and that `embeddings` is a list of objects that can be converted to `Embedding` dataclass
    # TODO: check if any embedding is missing
    embeddings = [
        Embedding(embedding=e.embedding, index=e.index) for e in api_response.data
    ]
    usage = Usage(
        prompt_tokens=api_response.usage.prompt_tokens,
        total_tokens=api_response.usage.total_tokens,
    )  # Assuming `usage` is an object with a `count` attribute

    # Assuming the model name is part of the response or set statically here
    model = api_response.model

    return EmbedderOutput(data=embeddings, model=model, usage=usage)


def process_images_for_response_api(
    images: Union[str, Dict, List[Union[str, Dict]]],
    encode_local_images: bool = True
) -> List[Dict[str, Any]]:
    """Process and validate images for OpenAI's responses.create API.
    
    This function handles various image input formats and converts them to the
    expected format for the responses.create API.
    
    Args:
        images: Can be:
            - A single image URL (str)
            - A single local file path (str)
            - A pre-formatted image dict with type='input_image'
            - A list containing any combination of the above
        encode_local_images: Whether to encode local image files to base64
        
    Returns:
        List of formatted image dicts ready for the API, each containing:
        - type: "input_image"
        - image_url: Either a URL or base64-encoded data URI
        
    Raises:
        ValueError: If image dict format is invalid
        FileNotFoundError: If local image file doesn't exist
        
    Examples:
        >>> # Single URL
        >>> process_images_for_response_api("https://example.com/image.jpg")
        [{"type": "input_image", "image_url": "https://example.com/image.jpg"}]
        
        >>> # Local file
        >>> process_images_for_response_api("/path/to/image.jpg")
        [{"type": "input_image", "image_url": "data:image/jpeg;base64,..."}]
        
        >>> # Multiple mixed sources
        >>> process_images_for_response_api([
        ...     "https://example.com/img.jpg",
        ...     "/local/img.png",
        ...     {"type": "input_image", "image_url": "..."}
        ... ])
        [...]
    """
    # Normalize input to list for uniform processing
    if not isinstance(images, list):
        images = [images]
    
    processed_images = []
    
    for img in images:
        if isinstance(img, str):
            if img.startswith(("http://", "https://")):
                # URL image
                processed_images.append({
                    "type": "input_image",
                    "image_url": img
                })
            elif img.startswith("data:"):
                # Data URI (already base64 encoded)
                processed_images.append({
                    "type": "input_image",
                    "image_url": img
                })
            else:
                # Local file path
                if encode_local_images:
                    if not os.path.isfile(img):
                        raise FileNotFoundError(f"Image file not found: {img}")
                    
                    # Encode to base64
                    base64_image = encode_image(img)
                    
                    # Determine MIME type from extension
                    ext = os.path.splitext(img)[1].lower()
                    mime_type = {
                        '.jpg': 'jpeg',
                        '.jpeg': 'jpeg',
                        '.png': 'png',
                        '.gif': 'gif',
                        '.webp': 'webp'
                    }.get(ext, 'jpeg')  # Default to jpeg
                    
                    processed_images.append({
                        "type": "input_image",
                        "image_url": f"data:image/{mime_type};base64,{base64_image}"
                    })
                else:
                    # Just pass the file path (for cases where encoding is handled elsewhere)
                    processed_images.append({
                        "type": "input_image",
                        "image_url": img
                    })
                    
        elif isinstance(img, dict):
            # Validate pre-formatted image dict
            if "type" not in img:
                raise ValueError(
                    f"Image dict must have 'type' field. Got: {img}"
                )
            if img["type"] != "input_image":
                raise ValueError(
                    f"Image dict must have type='input_image'. Got type='{img['type']}'"
                )
            if "image_url" not in img:
                raise ValueError(
                    f"Image dict must have 'image_url' field. Got: {img}"
                )
            
            # Valid format, add as-is
            processed_images.append(img)
            
        else:
            raise TypeError(
                f"Invalid image type: {type(img)}. "
                "Expected str (URL/path) or dict with type='input_image'"
            )
    
    return processed_images


def format_content_for_response_api(
    text: str,
    images: Optional[Union[str, Dict, List[Union[str, Dict]]]] = None
) -> List[Dict[str, Any]]:
    """Format text and optional images into content array for responses.create API.
    
    Args:
        text: The text prompt/question
        images: Optional images in various formats (see process_images_for_response_api)
        
    Returns:
        List of content items formatted for the API
        
    Examples:
        >>> # Text only
        >>> format_content_for_response_api("What is this?")
        [{"type": "input_text", "text": "What is this?"}]
        
        >>> # Text with image
        >>> format_content_for_response_api(
        ...     "What's in this image?",
        ...     "https://example.com/img.jpg"
        ... )
        [
            {"type": "input_text", "text": "What's in this image?"},
            {"type": "input_image", "image_url": "https://example.com/img.jpg"}
        ]
    """
    content = []
    
    # Add text content
    content.append({
        "type": "input_text",
        "text": text
    })
    
    # Add images if provided
    if images:
        image_content = process_images_for_response_api(images)
        content.extend(image_content)
    
    return content


def extract_text_from_response_stream(event) -> Optional[str]:
    """Extract text content from OpenAI Response API streaming events.
    
    The Response API generates various event types during streaming:
    - ResponseCreatedEvent: Initial event when response starts
    - ResponseInProgressEvent: Status updates
    - ResponseOutputItemAddedEvent: When new output items are added
    - ResponseContentPartAddedEvent: When content parts are added
    - ResponseTextDeltaEvent: Contains actual text chunks
    - ResponseTextDoneEvent: When text generation completes
    - ResponseDoneEvent: Final event when response completes
    
    This function extracts text only from ResponseTextDeltaEvent types.
    
    Args:
        event: A streaming event from OpenAI's Response API
        
    Returns:
        str: The text delta if this is a text delta event, None otherwise
        
    Examples:
        >>> # In a streaming response handler
        >>> async for event in stream:
        ...     text = extract_text_from_response_stream(event)
        ...     if text:
        ...         print(text, end="", flush=True)
    """
    # Check if the event has a type attribute indicating it's a text delta
    if hasattr(event, 'type') and event.type == 'response.output_text.delta':
        # ResponseTextDeltaEvent has a 'delta' field with the text chunk
        if hasattr(event, 'delta'):
            return event.delta
    
    return None


def extract_complete_text_from_response_stream(event) -> Optional[str]:
    """Extract complete text from response completion events.
    
    Some events contain the complete text rather than deltas:
    - ResponseTextDoneEvent: Contains the complete text when done
    - Response objects with output_text property
    
    Args:
        event: A streaming event from OpenAI's Response API
        
    Returns:
        str: The complete text if available, None otherwise
    """
    # Check for done events with complete text
    if hasattr(event, 'type') and event.type == 'response.output_text.done':
        if hasattr(event, 'text'):
            return event.text
    
    # Check for Response objects with output_text
    if hasattr(event, 'output_text') and not hasattr(event, 'delta'):
        return event.output_text
    
    return None


def is_response_complete(event) -> bool:
    """Check if a streaming event indicates the response is complete.
    
    Args:
        event: A streaming event from OpenAI's Response API
        
    Returns:
        bool: True if this event indicates completion, False otherwise
    """
    if hasattr(event, 'type'):
        return event.type in [
            'response.done',
            'response.output_text.done',
            'response.complete'
        ]
    
    if hasattr(event, 'status'):
        return event.status == 'completed'
    
    return False
