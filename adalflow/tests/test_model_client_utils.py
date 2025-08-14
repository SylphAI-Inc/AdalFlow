"""Tests for model client utility functions."""

import pytest
from unittest.mock import patch, mock_open
import os

from adalflow.components.model_client.utils import (
    process_images_for_response_api,
    format_content_for_response_api,
)


class TestProcessImagesForResponseAPI:
    """Test suite for process_images_for_response_api function."""
    
    def test_single_url_image(self):
        """Test processing a single URL image."""
        url = "https://example.com/image.jpg"
        result = process_images_for_response_api(url)
        
        assert len(result) == 1
        assert result[0] == {
            "type": "input_image",
            "image_url": url
        }
    
    def test_multiple_url_images(self):
        """Test processing multiple URL images."""
        urls = [
            "https://example.com/image1.jpg",
            "http://example.com/image2.png"
        ]
        result = process_images_for_response_api(urls)
        
        assert len(result) == 2
        for i, url in enumerate(urls):
            assert result[i] == {
                "type": "input_image",
                "image_url": url
            }
    
    @patch('adalflow.components.model_client.utils.os.path.isfile')
    @patch('adalflow.components.model_client.utils.encode_image')
    def test_local_image_jpeg(self, mock_encode, mock_isfile):
        """Test processing a local JPEG image."""
        mock_isfile.return_value = True
        mock_encode.return_value = "base64encodeddata"
        
        path = "/path/to/image.jpg"
        result = process_images_for_response_api(path)
        
        assert len(result) == 1
        assert result[0] == {
            "type": "input_image",
            "image_url": "data:image/jpeg;base64,base64encodeddata"
        }
        mock_isfile.assert_called_once_with(path)
        mock_encode.assert_called_once_with(path)
    
    @patch('adalflow.components.model_client.utils.os.path.isfile')
    @patch('adalflow.components.model_client.utils.encode_image')
    def test_local_image_png(self, mock_encode, mock_isfile):
        """Test processing a local PNG image."""
        mock_isfile.return_value = True
        mock_encode.return_value = "base64encodeddata"
        
        path = "/path/to/image.png"
        result = process_images_for_response_api(path)
        
        assert len(result) == 1
        assert result[0] == {
            "type": "input_image",
            "image_url": "data:image/png;base64,base64encodeddata"
        }
    
    @patch('adalflow.components.model_client.utils.os.path.isfile')
    @patch('adalflow.components.model_client.utils.encode_image')
    def test_local_image_webp(self, mock_encode, mock_isfile):
        """Test processing a local WebP image."""
        mock_isfile.return_value = True
        mock_encode.return_value = "base64encodeddata"
        
        path = "/path/to/image.webp"
        result = process_images_for_response_api(path)
        
        assert len(result) == 1
        assert result[0] == {
            "type": "input_image",
            "image_url": "data:image/webp;base64,base64encodeddata"
        }
    
    @patch('adalflow.components.model_client.utils.os.path.isfile')
    def test_local_image_not_found(self, mock_isfile):
        """Test error when local image file doesn't exist."""
        mock_isfile.return_value = False
        
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            process_images_for_response_api("/nonexistent/image.jpg")
    
    def test_preformatted_dict_valid(self):
        """Test processing a valid pre-formatted image dict."""
        image_dict = {
            "type": "input_image",
            "image_url": "data:image/jpeg;base64,/9j/4AAQ..."
        }
        result = process_images_for_response_api(image_dict)
        
        assert len(result) == 1
        assert result[0] == image_dict
    
    def test_preformatted_dict_missing_type(self):
        """Test error when dict is missing 'type' field."""
        image_dict = {
            "image_url": "https://example.com/image.jpg"
        }
        
        with pytest.raises(ValueError, match="Image dict must have 'type' field"):
            process_images_for_response_api(image_dict)
    
    def test_preformatted_dict_wrong_type(self):
        """Test error when dict has wrong 'type' value."""
        image_dict = {
            "type": "image_url",
            "image_url": "https://example.com/image.jpg"
        }
        
        with pytest.raises(ValueError, match="Image dict must have type='input_image'"):
            process_images_for_response_api(image_dict)
    
    def test_preformatted_dict_missing_url(self):
        """Test error when dict is missing 'image_url' field."""
        image_dict = {
            "type": "input_image"
        }
        
        with pytest.raises(ValueError, match="Image dict must have 'image_url' field"):
            process_images_for_response_api(image_dict)
    
    @patch('adalflow.components.model_client.utils.os.path.isfile')
    @patch('adalflow.components.model_client.utils.encode_image')
    def test_mixed_image_sources(self, mock_encode, mock_isfile):
        """Test processing mixed image sources."""
        mock_isfile.return_value = True
        mock_encode.return_value = "base64data"
        
        images = [
            "https://example.com/url_image.jpg",
            "/local/path/image.png",
            {
                "type": "input_image",
                "image_url": "data:image/gif;base64,R0lGOD..."
            }
        ]
        
        result = process_images_for_response_api(images)
        
        assert len(result) == 3
        # URL image
        assert result[0] == {
            "type": "input_image",
            "image_url": "https://example.com/url_image.jpg"
        }
        # Local image (encoded)
        assert result[1] == {
            "type": "input_image",
            "image_url": "data:image/png;base64,base64data"
        }
        # Pre-formatted dict
        assert result[2] == images[2]
    
    def test_invalid_image_type(self):
        """Test error when image is neither string nor dict."""
        with pytest.raises(TypeError, match="Invalid image type"):
            process_images_for_response_api(123)
        
        with pytest.raises(TypeError, match="Invalid image type"):
            process_images_for_response_api([None])
    
    @patch('adalflow.components.model_client.utils.os.path.isfile')
    def test_local_image_without_encoding(self, mock_isfile):
        """Test processing local image without encoding."""
        mock_isfile.return_value = True
        
        path = "/path/to/image.jpg"
        result = process_images_for_response_api(path, encode_local_images=False)
        
        assert len(result) == 1
        assert result[0] == {
            "type": "input_image",
            "image_url": path
        }


class TestFormatContentForResponseAPI:
    """Test suite for format_content_for_response_api function."""
    
    def test_text_only(self):
        """Test formatting text without images."""
        text = "What is the capital of France?"
        result = format_content_for_response_api(text)
        
        assert len(result) == 1
        assert result[0] == {
            "type": "input_text",
            "text": text
        }
    
    def test_text_with_single_url_image(self):
        """Test formatting text with a single URL image."""
        text = "What's in this image?"
        image = "https://example.com/photo.jpg"
        result = format_content_for_response_api(text, image)
        
        assert len(result) == 2
        assert result[0] == {
            "type": "input_text",
            "text": text
        }
        assert result[1] == {
            "type": "input_image",
            "image_url": image
        }
    
    def test_text_with_multiple_images(self):
        """Test formatting text with multiple images."""
        text = "Compare these images"
        images = [
            "https://example.com/img1.jpg",
            "https://example.com/img2.jpg"
        ]
        result = format_content_for_response_api(text, images)
        
        assert len(result) == 3
        assert result[0]["type"] == "input_text"
        assert result[0]["text"] == text
        for i, img_url in enumerate(images):
            assert result[i + 1] == {
                "type": "input_image",
                "image_url": img_url
            }
    
    def test_text_with_preformatted_image(self):
        """Test formatting text with a pre-formatted image dict."""
        text = "Analyze this"
        image_dict = {
            "type": "input_image",
            "image_url": "data:image/jpeg;base64,/9j/..."
        }
        result = format_content_for_response_api(text, image_dict)
        
        assert len(result) == 2
        assert result[0] == {
            "type": "input_text",
            "text": text
        }
        assert result[1] == image_dict
    
    @patch('adalflow.components.model_client.utils.os.path.isfile')
    @patch('adalflow.components.model_client.utils.encode_image')
    def test_text_with_local_image(self, mock_encode, mock_isfile):
        """Test formatting text with a local image file."""
        mock_isfile.return_value = True
        mock_encode.return_value = "base64encodeddata"
        
        text = "What object is this?"
        image_path = "/path/to/object.jpg"
        result = format_content_for_response_api(text, image_path)
        
        assert len(result) == 2
        assert result[0] == {
            "type": "input_text",
            "text": text
        }
        assert result[1] == {
            "type": "input_image",
            "image_url": "data:image/jpeg;base64,base64encodeddata"
        }
    
    def test_empty_text_with_image(self):
        """Test formatting empty text with image."""
        result = format_content_for_response_api("", "https://example.com/img.jpg")
        
        assert len(result) == 2
        assert result[0] == {
            "type": "input_text",
            "text": ""
        }
        assert result[1]["type"] == "input_image"
    
    def test_text_with_none_images(self):
        """Test formatting text with None as images parameter."""
        text = "Just text"
        result = format_content_for_response_api(text, None)
        
        assert len(result) == 1
        assert result[0] == {
            "type": "input_text",
            "text": text
        }
    
    def test_text_with_empty_image_list(self):
        """Test formatting text with empty image list."""
        text = "No images"
        result = format_content_for_response_api(text, [])
        
        assert len(result) == 1
        assert result[0] == {
            "type": "input_text",
            "text": text
        }


class TestIntegration:
    """Integration tests for realistic usage scenarios."""
    
    @patch('adalflow.components.model_client.utils.os.path.isfile')
    @patch('adalflow.components.model_client.utils.encode_image')
    def test_complete_multimodal_request(self, mock_encode, mock_isfile):
        """Test a complete multimodal request with mixed image sources."""
        mock_isfile.return_value = True
        mock_encode.return_value = "localimagebase64"
        
        text = "Compare these three images and describe the differences"
        images = [
            "https://example.com/web_image.jpg",
            "/local/path/to/image.png",
            {
                "type": "input_image",
                "image_url": "data:image/gif;base64,R0lGODlhAQABAIAAAP..."
            }
        ]
        
        result = format_content_for_response_api(text, images)
        
        # Check structure
        assert len(result) == 4  # 1 text + 3 images
        
        # Check text
        assert result[0]["type"] == "input_text"
        assert result[0]["text"] == text
        
        # Check URL image
        assert result[1]["type"] == "input_image"
        assert result[1]["image_url"] == "https://example.com/web_image.jpg"
        
        # Check local image (should be encoded)
        assert result[2]["type"] == "input_image"
        assert result[2]["image_url"] == "data:image/png;base64,localimagebase64"
        
        # Check pre-formatted image
        assert result[3] == images[2]
    
    def test_api_format_compatibility(self):
        """Test that output format matches OpenAI API expectations."""
        text = "What's in this image?"
        image = "https://example.com/test.jpg"
        
        content = format_content_for_response_api(text, image)
        
        # This should be ready to wrap in the API format:
        api_input = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        # Verify structure
        assert api_input[0]["role"] == "user"
        assert isinstance(api_input[0]["content"], list)
        assert len(api_input[0]["content"]) == 2
        
        # Verify content items
        assert api_input[0]["content"][0]["type"] == "input_text"
        assert api_input[0]["content"][1]["type"] == "input_image"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])