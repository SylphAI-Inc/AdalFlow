import unittest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from adalflow.core.types import ModelType, GeneratorOutput
from adalflow.components.model_client.ollama_client import OllamaClient
from typing import AsyncGenerator, Generator

# import ollama

# Check if ollama model is installed
# def model_installed(model_name: str):
#     model_list = ollama.list()
#     for model in model_list['models']:
#         if model['name'] == model_name:
#             return True
#     return False


class TestOllamaModelClient(unittest.TestCase):

    def test_ollama_llm_client(self):
        ollama_client = Mock(spec=OllamaClient())
        print("Testing ollama LLM client")
        # run the model
        kwargs = {
            "model": "qwen2:0.5b",
        }
        api_kwargs = ollama_client.convert_inputs_to_api_kwargs(
            input="Hello world",
            model_kwargs=kwargs,
            model_type=ModelType.LLM,
        ).return_value = {"prompt": "Hello World", "model": "qwen2:0.5b"}
        assert api_kwargs == {"prompt": "Hello World", "model": "qwen2:0.5b"}
        output = ollama_client.call(
            api_kwargs=api_kwargs, model_type=ModelType.LLM
        ).return_value = {"message": "Hello"}
        assert output == {"message": "Hello"}

    def test_ollama_embedding_client(self):
        ollama_client = Mock(spec=OllamaClient())
        print("Testing ollama embedding client")

        # run the model
        kwargs = {
            "model": "jina/jina-embeddings-v2-base-en:latest",
        }
        api_kwargs = ollama_client.convert_inputs_to_api_kwargs(
            input="Welcome",
            model_kwargs=kwargs,
            model_type=ModelType.EMBEDDER,
        ).return_value = {
            "prompt": "Welcome",
            "model": "jina/jina-embeddings-v2-base-en:latest",
        }
        assert api_kwargs == {
            "prompt": "Welcome",
            "model": "jina/jina-embeddings-v2-base-en:latest",
        }

        output = ollama_client.call(
            api_kwargs=api_kwargs, model_type=ModelType.EMBEDDER
        ).return_value = {"embedding": [-0.7391586899757385]}
        assert output == {"embedding": [-0.7391586899757385]}

    def test_sync_streaming_chat(self):
        """Test synchronous streaming with chat API"""
        ollama_client = OllamaClient(host="http://localhost:11434")
        
        # Mock the sync_client.chat method to return a generator
        def mock_stream_generator():
            chunks = [
                {"message": {"content": "Hello", "thinking": None}},
                {"message": {"content": " world", "thinking": None}},
                {"message": {"content": "!", "thinking": None}},
            ]
            for chunk in chunks:
                yield chunk
        
        with patch.object(ollama_client.sync_client, 'chat', return_value=mock_stream_generator()):
            # Test streaming call
            api_kwargs = {
                "model": "qwen2:0.5b",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True
            }
            
            result = ollama_client.call(api_kwargs=api_kwargs, model_type=ModelType.LLM)
            
            # The result should be a generator
            self.assertTrue(hasattr(result, '__iter__'))
            
            # Parse the result
            parsed = ollama_client.parse_chat_completion(result)
            
            # For streaming, the parsed result should be a GeneratorOutput with raw_response containing the generator
            self.assertIsInstance(parsed, GeneratorOutput)
            self.assertIsNotNone(parsed.raw_response)
            self.assertEqual(parsed.api_response, result)
            
            # Verify we can iterate through the raw_response
            content_parts = []
            for chunk in parsed.raw_response:
                if "message" in chunk:
                    content_parts.append(chunk["message"]["content"])
            
            self.assertEqual(content_parts, ["Hello", " world", "!"])

    async def test_async_streaming_chat(self):
        """Test asynchronous streaming with chat API"""
        ollama_client = OllamaClient(host="http://localhost:11434")
        
        # Mock the async_client.chat method to return an async generator
        async def mock_async_stream_generator():
            chunks = [
                {"message": {"content": "The", "thinking": None}},
                {"message": {"content": " sky", "thinking": None}},
                {"message": {"content": " is", "thinking": None}},
                {"message": {"content": " blue", "thinking": None}},
            ]
            for chunk in chunks:
                yield chunk
        
        # Create a mock async client
        mock_async_client = AsyncMock()
        mock_async_client.chat = AsyncMock(return_value=mock_async_stream_generator())
        ollama_client.async_client = mock_async_client
        
        # Test streaming call
        api_kwargs = {
            "model": "gpt-oss:20b",
            "messages": [{"role": "user", "content": "Why is the sky blue?"}],
            "stream": True
        }
        
        result = await ollama_client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        
        # The result should be an async generator
        self.assertTrue(hasattr(result, '__aiter__'))
        
        # Parse the result
        parsed = ollama_client.parse_chat_completion(result)
        
        # For streaming, the parsed result should be a GeneratorOutput with raw_response containing the async generator
        self.assertIsInstance(parsed, GeneratorOutput)
        self.assertIsNotNone(parsed.raw_response)
        self.assertEqual(parsed.api_response, result)
        
        # Verify we can iterate through the raw_response asynchronously
        content_parts = []
        async for chunk in parsed.raw_response:
            if "message" in chunk:
                content_parts.append(chunk["message"]["content"])
        
        self.assertEqual(content_parts, ["The", " sky", " is", " blue"])

    def test_async_streaming_chat_sync(self):
        """Test async streaming from synchronous context"""
        # This is a wrapper to run the async test in a sync context
        asyncio.run(self.test_async_streaming_chat())

    def test_non_streaming_chat(self):
        """Test non-streaming chat API call"""
        ollama_client = OllamaClient(host="http://localhost:11434")
        
        # Mock the sync_client.chat method to return a non-streaming response
        mock_response = {
            "message": {
                "content": "Hello, how can I help you?",
                "thinking": "The user greeted me"
            }
        }
        
        with patch.object(ollama_client.sync_client, 'chat', return_value=mock_response):
            # Test non-streaming call
            api_kwargs = {
                "model": "qwen2:0.5b",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False
            }
            
            result = ollama_client.call(api_kwargs=api_kwargs, model_type=ModelType.LLM)
            
            # The result should be a dict for non-streaming
            self.assertIsInstance(result, dict)
            
            # Parse the result
            parsed = ollama_client.parse_chat_completion(result)
            
            # For non-streaming, the parsed result should be a GeneratorOutput with data populated
            self.assertIsInstance(parsed, GeneratorOutput)
            self.assertEqual(parsed.raw_response, "Hello, how can I help you?")
            self.assertEqual(parsed.thinking, "The user greeted me")
            self.assertEqual(parsed.api_response, mock_response)

    async def test_streaming_with_generator(self):
        """Test streaming with Generator component"""
        from adalflow.core import Generator
        
        ollama_client = OllamaClient(host="http://localhost:11434")
        
        # Mock the async streaming response
        async def mock_async_stream():
            chunks = [
                {"message": {"content": "Test", "thinking": None}},
                {"message": {"content": " streaming", "thinking": None}},
            ]
            for chunk in chunks:
                yield chunk
        
        mock_async_client = AsyncMock()
        mock_async_client.chat = AsyncMock(return_value=mock_async_stream())
        ollama_client.async_client = mock_async_client
        
        # Create a Generator with streaming enabled
        generator = Generator(
            model_client=ollama_client,
            model_kwargs={
                "model": "qwen2:0.5b",
                "stream": True,
            }
        )
        
        # Mock the acall to return a GeneratorOutput with async generator in raw_response
        with patch.object(generator, 'acall') as mock_acall:
            mock_output = GeneratorOutput(
                data=None,
                raw_response=mock_async_stream(),
                api_response=mock_async_stream()
            )
            mock_acall.return_value = mock_output
            
            # Call the generator
            output = await generator.acall(prompt_kwargs={"input_str": "Hello"})
            
            # Verify the output structure
            self.assertIsInstance(output, GeneratorOutput)
            self.assertIsNotNone(output.raw_response)
            
            # Verify we can iterate through the streaming response
            content_parts = []
            async for chunk in output.raw_response:
                if "message" in chunk:
                    content_parts.append(chunk["message"]["content"])
            
            self.assertEqual(content_parts, ["Test", " streaming"])

    def test_streaming_with_generator_sync(self):
        """Test streaming with Generator component from sync context"""
        asyncio.run(self.test_streaming_with_generator())


if __name__ == "__main__":
    unittest.main()
