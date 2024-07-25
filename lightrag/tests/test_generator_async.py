import pytest
from typing import AsyncGenerator
from unittest.mock import AsyncMock
from lightrag.components.model_client.ollama_client import OllamaClient
from lightrag.core import Generator

async def async_gen() -> AsyncGenerator[int, None]:
    yield {"response": "I"}
    yield {"response": " am"}
    yield {"response": " hungry"}


@pytest.mark.asyncio
async def test_acall():    
    ollama_client = OllamaClient()
    ollama_client.acall = AsyncMock(return_value = async_gen())
    
    generator = Generator(model_client=ollama_client)
    output = await generator.acall({}, {})

    result = []
    async for value in output.data:
        result.append(value)
    assert result == ["I", " am", " hungry"]

