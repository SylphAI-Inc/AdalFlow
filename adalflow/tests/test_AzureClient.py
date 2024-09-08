import unittest
from unittest.mock import patch, MagicMock
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
from openai.types import Completion, CreateEmbeddingResponse
from openai.types.chat import ChatCompletionChunk, ChatCompletion
from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, EmbedderOutput, TokenLogProb, CompletionUsage, GeneratorOutput
from adalflow.components.model_client.openai_client import AzureAIClient  

class TestAzureAIClient(unittest.TestCase):

    @patch('adalflow.components.model_client.openai_client.AzureOpenAI')  
    @patch('adalflow.components.model_client.openai_client.DefaultAzureCredential')
    def setUp(self, MockDefaultAzureCredential, MockAzureOpenAI):
        self.mock_credential = MockDefaultAzureCredential()
        self.mock_sync_client = MockAzureOpenAI.return_value
        self.client = AzureAIClient(
            api_key="test_api_key",
            api_version="v1",
            azure_endpoint="https://test.endpoint",
            credential=self.mock_credential,
        )
        self.client.sync_client = self.mock_sync_client

    def test_init_sync_client_with_api_key(self):
        client = AzureAIClient(api_key="test_key", api_version="v1", azure_endpoint="https://test.endpoint")
        self.assertIsInstance(client.sync_client, AzureOpenAI)

    def test_init_sync_client_with_credential(self):
        client = AzureAIClient(
            api_version="v1",
            azure_endpoint="https://test.endpoint",
            credential=self.mock_credential
        )
        self.assertIsInstance(client.sync_client, AzureOpenAI)

    @patch('adalflow.components.model_client.openai_client.AzureOpenAI')
    def test_call_embeddings(self, MockAzureOpenAI):
        mock_embeddings = MagicMock()
        MockAzureOpenAI.return_value.embeddings.create = mock_embeddings
        api_kwargs = {'input': ["test"]}
        model_type = ModelType.EMBEDDER
        self.client.call(api_kwargs=api_kwargs, model_type=model_type)
        MockAzureOpenAI.return_value.embeddings.create.assert_called_once_with(**api_kwargs)

    @patch('adalflow.components.model_client.openai_client.AzureOpenAI')
    def test_call_chat_completions(self, MockAzureOpenAI):
        mock_chat_completions = MagicMock()
        MockAzureOpenAI.return_value.chat.completions.create = mock_chat_completions
        api_kwargs = {'input': "test"}
        model_type = ModelType.LLM
        self.client.call(api_kwargs=api_kwargs, model_type=model_type)
        MockAzureOpenAI.return_value.chat.completions.create.assert_called_once_with(**api_kwargs)

    @patch('adalflow.components.model_client.openai_client.AzureOpenAI')
    def test_parse_chat_completion(self, MockAzureOpenAI):
        mock_chat_completion = MagicMock(spec=ChatCompletion)
        mock_chat_completion.choices = [MagicMock(message=MagicMock(content="test_content"))]
        self.client.chat_completion_parser = lambda completion: completion.choices[0].message.content
        result = self.client.parse_chat_completion(mock_chat_completion)
        self.assertEqual(result, "test_content")

    @patch('adalflow.components.model_client.openai_client.AzureOpenAI')
    def test_track_completion_usage(self, MockAzureOpenAI):
        mock_chat_completion = MagicMock(spec=ChatCompletion)
        mock_chat_completion.usage = MagicMock(
            completion_tokens=10,
            prompt_tokens=5,
            total_tokens=15
        )
        result = self.client.track_completion_usage(mock_chat_completion)
        self.assertEqual(result.completion_tokens, 10)
        self.assertEqual(result.prompt_tokens, 5)
        self.assertEqual(result.total_tokens, 15)

    @patch('adalflow.components.model_client.openai_client.AzureOpenAI')
    def test_parse_embedding_response(self, MockAzureOpenAI):
        mock_embedding_response = MagicMock(spec=CreateEmbeddingResponse)
        self.client.parse_embedding_response = lambda response: EmbedderOutput(data=["test_embedding"], error=None, raw_response=response)
        result = self.client.parse_embedding_response(mock_embedding_response)
        self.assertEqual(result.data, ["test_embedding"])

    @patch('adalflow.components.model_client.openai_client.AzureOpenAI')
    def test_convert_inputs_to_api_kwargs(self, MockAzureOpenAI):
        input_data = "test input"
        model_kwargs = {"param": "value"}
        result = self.client.convert_inputs_to_api_kwargs(input=input_data, model_kwargs=model_kwargs, model_type=ModelType.LLM)
        expected = {
            "input": input_data,
            "param": "value"
        }
        self.assertEqual(result, expected)

    def test_from_dict(self):
        data = {
            "api_key": "test_key",
            "api_version": "v1",
            "azure_endpoint": "https://test.endpoint",
            "credential": self.mock_credential,
        }
        client = AzureAIClient.from_dict(data)
        self.assertEqual(client._api_key, "test_key")
        self.assertEqual(client._apiversion, "v1")
        self.assertEqual(client._azure_endpoint, "https://test.endpoint")

    def test_to_dict(self):
        expected = {
            "api_key": "test_api_key",
            "api_version": "v1",
            "azure_endpoint": "https://test.endpoint",
            "credential": self.mock_credential,
        }
        result = self.client.to_dict()
        for key, value in expected.items():
            self.assertEqual(result.get(key), value)

if __name__ == '__main__':
    unittest.main()
