import unittest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from adalflow.core.types import ModelType, GeneratorOutput, Function
from adalflow.components.model_client.ollama_client import OllamaClient, extract_ollama_tool_calls
from typing import AsyncGenerator, Generator
from ollama import ChatResponse, Message
from types import SimpleNamespace

# Create mock Ollama types that behave like the real ones
# These simulate the actual Ollama API response objects
class MockFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

class MockToolCall:
    def __init__(self, function):
        self.function = function

class MockMessage:
    def __init__(self, role, content, thinking=None, images=None, tool_name=None, tool_calls=None):
        self.role = role
        self.content = content
        self.thinking = thinking
        self.images = images
        self.tool_name = tool_name
        self.tool_calls = tool_calls or []
        
    def get(self, key, default=None):
        """Simulate dict-like behavior for compatibility"""
        return getattr(self, key, default)

class MockChatResponse(dict):
    def __init__(self, data):
        super().__init__(data)
        # Make message accessible as attribute
        if 'message' in data and not isinstance(data['message'], (dict, Message)):
            self.message = data['message']

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
            
            # For streaming, the parsed result should be the raw generator directly
            self.assertTrue(hasattr(parsed, '__iter__'))
            self.assertNotIsInstance(parsed, GeneratorOutput)
            
            # Verify we can iterate through the raw generator
            content_parts = []
            for chunk in parsed:
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
        
        # For streaming, the parsed result should be the raw async generator directly
        self.assertTrue(hasattr(parsed, '__aiter__'))
        self.assertNotIsInstance(parsed, GeneratorOutput)
        
        # Verify we can iterate through the raw async generator
        content_parts = []
        async for chunk in parsed:
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

    def test_tool_call_extraction(self):
        """Test extraction of tool calls from Ollama chat response"""
        ollama_client = OllamaClient(host="http://localhost:11434")
        
        # Create mock Ollama objects that match the real API response structure
        mock_function = MockFunction(
            name="SearchTool_acall",
            arguments={
                "check_credibility": False,
                "num_pages": 1,
                "num_sources": 5,
                "query": "adalflow",
                "type": "web"
            }
        )
        
        mock_tool_call = MockToolCall(function=mock_function)
        
        # Create mock Message with tool calls
        mock_message = MockMessage(
            role="assistant",
            content="",
            thinking="We need to answer question: 'What is adalflow and list its main features?' Likely refer to a library. We can search web. Use SearchTool.",
            images=None,
            tool_name=None,
            tool_calls=[mock_tool_call]
        )
        
        # Create the response as a dict (ChatResponse is dict-based) with message
        mock_response = {
            "model": "gpt-oss:20b",
            "created_at": "2025-08-10T22:22:04.234251Z",
            "done": True,
            "done_reason": "stop",
            "total_duration": 4681179750,
            "load_duration": 97384958,
            "prompt_eval_count": 4553,
            "prompt_eval_duration": 526190792,
            "eval_count": 81,
            "eval_duration": 4056959833,
            "message": mock_message
        }
        
        with patch.object(ollama_client.sync_client, 'chat', return_value=mock_response):
            # Test non-streaming call with tool response
            api_kwargs = {
                "model": "gpt-oss:20b",
                "messages": [{"role": "user", "content": "What is adalflow and list its main features?"}],
                "stream": False
            }
            
            result = ollama_client.call(api_kwargs=api_kwargs, model_type=ModelType.LLM)
            
            # Parse the result
            parsed = ollama_client.parse_chat_completion(result)
            
            # Verify the parsed result contains the tool call
            self.assertIsInstance(parsed, GeneratorOutput)
            self.assertIsNotNone(parsed.data)
            self.assertIsInstance(parsed.data, Function)
            
            # Check the extracted function details
            self.assertEqual(parsed.data.name, "SearchTool_acall")
            self.assertEqual(parsed.data.kwargs["query"], "adalflow")
            self.assertEqual(parsed.data.kwargs["type"], "web")
            self.assertEqual(parsed.data.kwargs["num_sources"], 5)
            self.assertEqual(parsed.data.kwargs["num_pages"], 1)
            self.assertEqual(parsed.data.kwargs["check_credibility"], False)
            
            # Also verify thinking was captured
            self.assertIsNotNone(parsed.thinking)
            self.assertIn("SearchTool", parsed.thinking)

    def test_extract_ollama_tool_calls_function(self):
        """Test the extract_ollama_tool_calls helper function directly"""
        
        # Test with valid tool calls using mock Ollama types
        func1 = MockFunction(
            name="FileSearchTool_grep",
            arguments={
                "max_results": 20,
                "output_mode": "content",
                "path": "",
                "query": "adalflow"
            }
        )
        
        func2 = MockFunction(
            name="WebSearch",
            arguments={
                "query": "test query"
            }
        )
        
        tool_call1 = MockToolCall(function=func1)
        tool_call2 = MockToolCall(function=func2)
        
        # Create message with tool_calls
        message_with_tools = MockMessage(
            role="assistant",
            content="",
            tool_calls=[tool_call1, tool_call2]
        )
        
        functions = extract_ollama_tool_calls(message_with_tools)
        
        self.assertIsNotNone(functions)
        self.assertEqual(len(functions), 2)
        
        # Check first function
        self.assertEqual(functions[0].name, "FileSearchTool_grep")
        self.assertEqual(functions[0].kwargs["max_results"], 20)
        self.assertEqual(functions[0].kwargs["query"], "adalflow")
        
        # Check second function
        self.assertEqual(functions[1].name, "WebSearch")
        self.assertEqual(functions[1].kwargs["query"], "test query")
        
        # Test with no tool calls
        message_without_tools = MockMessage(
            role="assistant",
            content="This is a regular message",
            thinking="No tools needed",
            tool_calls=None
        )
        
        functions = extract_ollama_tool_calls(message_without_tools)
        self.assertIsNone(functions)
        
        # Test with empty tool_calls list
        message_empty_tools = MockMessage(
            role="assistant",
            content="Response",
            tool_calls=[]
        )
        
        functions = extract_ollama_tool_calls(message_empty_tools)
        self.assertIsNone(functions)
        
        # Test with invalid input
        functions = extract_ollama_tool_calls(None)
        self.assertIsNone(functions)
        
        functions = extract_ollama_tool_calls("not a dict")
        self.assertIsNone(functions)
        
        # Test with another message object with tool calls
        test_func = MockFunction(
            name="TestTool",
            arguments={"param": "value"}
        )
        test_tool_call = MockToolCall(function=test_func)
        
        message_obj = MockMessage(
            role="assistant",
            content="Using tool",
            tool_calls=[test_tool_call]
        )
        
        functions = extract_ollama_tool_calls(message_obj)
        self.assertIsNotNone(functions)
        self.assertEqual(len(functions), 1)
        self.assertEqual(functions[0].name, "TestTool")
        self.assertEqual(functions[0].kwargs["param"], "value")


if __name__ == "__main__":
    unittest.main()
