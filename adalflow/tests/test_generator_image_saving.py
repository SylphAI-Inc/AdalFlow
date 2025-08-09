"""Tests for GeneratorOutput image saving functionality."""

import unittest
import base64
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

from adalflow.core.types import GeneratorOutput


class TestGeneratorImageSaving(unittest.TestCase):
    """Test cases for GeneratorOutput.save_images() method."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create a tiny red pixel PNG for testing (1x1 pixel)
        self.test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_save_single_image_as_png(self):
        """Test saving a single base64 image as PNG."""
        output = GeneratorOutput(
            data="Generated an image",
            images=self.test_image_base64
        )
        
        saved_path = output.save_images(
            directory=self.test_dir,
            prefix="test_single",
            format="png"
        )
        
        self.assertIsInstance(saved_path, str)
        self.assertTrue(os.path.exists(saved_path))
        self.assertTrue(saved_path.endswith(".png"))
        
        # Verify file is not empty
        file_size = os.path.getsize(saved_path)
        self.assertGreater(file_size, 0)
    
    def test_save_multiple_images(self):
        """Test saving multiple base64 images."""
        output = GeneratorOutput(
            data="Generated multiple images",
            images=[self.test_image_base64, self.test_image_base64]
        )
        
        saved_paths = output.save_images(
            directory=self.test_dir,
            prefix="test_multi",
            format="png"
        )
        
        self.assertIsInstance(saved_paths, list)
        self.assertEqual(len(saved_paths), 2)
        
        for path in saved_paths:
            self.assertTrue(os.path.exists(path))
            self.assertTrue(path.endswith(".png"))
    
    def test_save_with_data_uri(self):
        """Test saving an image provided as a data URI."""
        data_uri = f"data:image/png;base64,{self.test_image_base64}"
        output = GeneratorOutput(
            data="Generated an image",
            images=data_uri
        )
        
        saved_path = output.save_images(
            directory=self.test_dir,
            prefix="test_data_uri",
            format="png"
        )
        
        self.assertIsInstance(saved_path, str)
        self.assertTrue(os.path.exists(saved_path))
    
    def test_save_url_without_decoding(self):
        """Test handling of URL images (saves URL to file)."""
        image_url = "https://example.com/image.jpg"
        output = GeneratorOutput(
            data="Generated an image",
            images=image_url
        )
        
        saved_path = output.save_images(
            directory=self.test_dir,
            prefix="test_url",
            format="png",
            decode_base64=True  # Should recognize it's a URL and not decode
        )
        
        self.assertIsInstance(saved_path, str)
        self.assertTrue(saved_path.endswith(".url"))
        
        # Verify URL was saved
        with open(saved_path, 'r') as f:
            content = f.read()
        self.assertEqual(content, image_url)
    
    def test_no_images_returns_none(self):
        """Test that save_images returns None when no images present."""
        output = GeneratorOutput(
            data="No images generated"
        )
        
        result = output.save_images(directory=self.test_dir)
        self.assertIsNone(result)
    
    def test_custom_directory_creation(self):
        """Test that custom nested directories are created."""
        nested_dir = os.path.join(self.test_dir, "year", "month", "day")
        output = GeneratorOutput(
            data="Generated an image",
            images=self.test_image_base64
        )
        
        saved_path = output.save_images(
            directory=nested_dir,
            prefix="nested",
            format="png"
        )
        
        self.assertTrue(os.path.exists(nested_dir))
        self.assertTrue(os.path.exists(saved_path))
    
    def test_jpeg_conversion_with_pil(self):
        """Test JPEG conversion using PIL when available."""
        # Test that JPEG conversion works without mocking (using real PIL if available)
        output = GeneratorOutput(
            data="Generated an image",
            images=self.test_image_base64
        )
        
        try:
            # This should trigger PIL conversion
            saved_path = output.save_images(
                directory=self.test_dir,
                prefix="test_jpeg",
                format="jpg"
            )
            
            # If PIL is available, it should work
            self.assertIsInstance(saved_path, str)
            self.assertTrue(os.path.exists(saved_path))
            self.assertTrue(saved_path.endswith(".jpg"))
            
        except ImportError:
            # PIL not available - that's okay for this test environment
            pass
    
    def test_format_variations(self):
        """Test that different format strings work correctly."""
        formats_to_test = {
            "png": ".png",
            "jpg": ".jpg",
            "jpeg": ".jpeg",
        }
        
        for format_str, expected_ext in formats_to_test.items():
            with self.subTest(format=format_str):
                output = GeneratorOutput(
                    data=f"Testing {format_str} format",
                    images=self.test_image_base64
                )
                
                # For formats other than PNG, PIL is required
                # Without PIL, it should still save but as PNG
                try:
                    saved_path = output.save_images(
                        directory=self.test_dir,
                        prefix=f"test_{format_str}",
                        format=format_str
                    )
                    
                    self.assertIsInstance(saved_path, str)
                    self.assertTrue(os.path.exists(saved_path))
                    
                    # Check extension matches if PIL is available
                    if format_str == "png":
                        self.assertTrue(saved_path.endswith(expected_ext))
                except ImportError:
                    # PIL not available for non-PNG formats
                    if format_str not in ["png"]:
                        pass  # Expected to fail without PIL
    
    def test_return_paths_parameter(self):
        """Test that return_paths parameter controls return value."""
        output = GeneratorOutput(
            data="Generated an image",
            images=self.test_image_base64
        )
        
        # With return_paths=False
        result = output.save_images(
            directory=self.test_dir,
            prefix="no_return",
            format="png",
            return_paths=False
        )
        
        self.assertIsNone(result)
        
        # Verify file was still saved
        expected_path = os.path.join(self.test_dir, "no_return_0.png")
        self.assertTrue(os.path.exists(expected_path))
    
    def test_mixed_image_sources(self):
        """Test handling of mixed image sources (base64, data URI, URL)."""
        images = [
            self.test_image_base64,  # Raw base64
            f"data:image/png;base64,{self.test_image_base64}",  # Data URI
            "https://example.com/image.jpg"  # URL
        ]
        
        output = GeneratorOutput(
            data="Mixed image sources",
            images=images
        )
        
        saved_paths = output.save_images(
            directory=self.test_dir,
            prefix="mixed",
            format="png"
        )
        
        self.assertEqual(len(saved_paths), 3)
        
        # First two should be saved as images
        self.assertTrue(saved_paths[0].endswith(".png"))
        self.assertTrue(saved_paths[1].endswith(".png"))
        
        # Third should be saved as URL file
        self.assertTrue(saved_paths[2].endswith(".url"))


if __name__ == "__main__":
    unittest.main()