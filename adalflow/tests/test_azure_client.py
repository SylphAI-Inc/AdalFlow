# tests/test_azure_ai_client.py

import os
import unittest
from unittest.mock import patch, MagicMock, ANY
from adalflow.components.model_client.azureai_client import AzureAIClient  # Adjust based on your project structure
from adalflow.core.types import ModelType
from openai import *
from openai.types import CreateEmbeddingResponse,CompletionUsage
from openai.types.chat import ChatCompletion
from openai import AzureOpenAI
class TestAzureAIClient(unittest.TestCase):
    def setUp(self):
        # Set environment variables for testing
        os.environ["AZURE_OPENAI_API_KEY"] = "test_api_key"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test-endpoint.openai.azure.com/"
        os.environ["AZURE_OPENAI_VERSION"] = "2023-05-15"

    def tearDown(self):
        # Clean up environment variables after tests
        del os.environ["AZURE_OPENAI_API_KEY"]
        del os.environ["AZURE_OPENAI_ENDPOINT"]
        del os.environ["AZURE_OPENAI_VERSION"]

    @patch('adalflow.components.model_client.azureai_client.AzureOpenAI', autospec=True)
    def test_init_with_api_key(self, mock_azure_openai):
        # Configure the mock to return a MagicMock instance when instantiated
        mock_instance = MagicMock()
        mock_azure_openai.return_value = mock_instance

        # Instantiate the client
        client = AzureAIClient()

        # Assert that AzureOpenAI was called with the correct parameters
        mock_azure_openai.assert_called_with(
            api_key="test_api_key",
            azure_endpoint="https://test-endpoint.openai.azure.com/",
            api_version="2023-05-15"
        )

        # Verify internal state of AzureAIClient
        self.assertIsNotNone(client.sync_client)
        self.assertIsNone(client.async_client)
        self.assertEqual(client.api_type, "azure")
        self.assertEqual(client._api_key, "test_api_key")
        self.assertEqual(client._apiversion, "2023-05-15")
        self.assertEqual(client._azure_endpoint, "https://test-endpoint.openai.azure.com/")
        self.assertIsNone(client._credential)
        
    
    @patch('adalflow.components.model_client.azureai_client.AzureOpenAI', autospec=True)
    def test_init_with_credentials(self, mock_azure_openai):
        # Arrange
        mock_instance = MagicMock()
        mock_azure_openai.return_value = mock_instance
        mock_credential = MagicMock()

        # Act
        client = AzureAIClient(credential=mock_credential)

        # Assert
        mock_azure_openai.assert_called_once_with(
            api_key="test_api_key",  # Using ANY as the exact object is complex
            azure_endpoint="https://test-endpoint.openai.azure.com/",
            api_version="2023-05-15"
        )
       
        self.assertEqual(client._credential, mock_credential)    
    
    
    @patch('adalflow.components.model_client.azureai_client.AzureOpenAI', autospec=True)
    def test_call_unsupported_model_type(self, mock_azure_openai):
        # Arrange
        client = AzureAIClient(api_key="test_api_key")

        api_kwargs = {
            "model": "unsupported-model",
            "messages": [{"role": "user", "content": "Hi"}]
        }
        
            
        # Act & Assert
        with self.assertRaises(ValueError) as context:    
            client.call(api_kwargs=api_kwargs, model_type=ModelType.UNDEFINED)
        
        self.assertIn("model_type ModelType.UNDEFINED is not supported", str(context.exception))
        # mock_azure_openai.assert_not_called()
    
    # Additional tests can follow the same pattern

if __name__ == '__main__':
    unittest.main()
