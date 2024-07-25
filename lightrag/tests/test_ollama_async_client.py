import pytest
from typing import AsyncGenerator
from unittest.mock import AsyncMock
from lightrag.core.types import ModelType
from lightrag.components.model_client.ollama_client import OllamaClient

@pytest.mark.asyncio
async def test_ollama_llm_client_async():
    ollama_client = AsyncMock(spec=OllamaClient())
    ollama_client.acall.return_value = {"message": "Hello"}
    print("Testing ollama LLM async client")
    
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

    output = await ollama_client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
    assert output == {"message": "Hello"}


async def async_gen() -> AsyncGenerator[int, None]:
    yield {"response": "I"}
    yield {"response": " am"}
    yield {"response": " cool"}

@pytest.mark.asyncio
async def test_async_generator_completion():
    ollama_client = OllamaClient()
    print("Testing ollama LLM async client")

    output = await ollama_client.aparse_chat_completion(async_gen())

    result = []
    async for value in output:
        result.append(value)
        
    assert result == ["I", " am", " cool"]
