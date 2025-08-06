import asyncio
from ollama import AsyncClient

async def chat():
    message = {'role': 'user', 'content': 'Why is the sky blue?'}
    async for part in await AsyncClient().chat(model='gpt-oss:20b', messages=[message], stream=True):
        print(part['message']['content'], end='', flush=True)

# asyncio.run(chat())

async def chat_non_streaming():
    message = {'role': 'user', 'content': 'Hi'}
    response = await AsyncClient().chat(model='gpt-oss:20b', messages=[message], stream=False)
    print(response['message'])

# asyncio.run(chat_non_streaming())

from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core import Generator 

gen = Generator(
    model_client=OllamaClient(host="http://localhost:11434"),
    model_kwargs={
        "model": "gpt-oss:20b",
        "options": {
            "temperature": 0.7,
            "num_predict": 512,
            "top_k": 40,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_ctx": 2048,
        }
    }
)

output = asyncio.run(gen.acall(prompt_kwargs={"input_str": "hi"}))
print(output)


# test call 
output = gen.call(prompt_kwargs={"input_str": "hi"})
print(output)