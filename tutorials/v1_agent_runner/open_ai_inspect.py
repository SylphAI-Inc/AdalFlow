import inspect
from agents import OpenAIChatCompletionsModel

print(inspect.getsource(OpenAIChatCompletionsModel))

from openai import OpenAI

client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "user",
            "content": "Say 'double bubble bath' ten times fast.",
        },
    ],
    stream=True,
)

for chunk in stream:
    print(chunk)
    print(chunk.choices[0].delta)
    print("****************")
