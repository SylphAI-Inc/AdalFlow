from lightrag.core.generator import Generator
from lightrag.core.component import Component
from lightrag.components.model_client.ollama_client import OllamaClient
from lightrag.utils import setup_env

from ollama import Client

setup_env()

client = Client(host="http://localhost:11434")
# response = client.chat(
#     model="llama3",
#     messages=[
#         {
#             "role": "user",
#             "content": "Why is the sky blue?",
#         },
#     ],
# )
# print(response)

# response = client.generate(
#     model="llama3",
#     prompt="Why is the sky blue?",
# )
# print(response)


# Create components that will serve as function calls to our local LLM
class SimpleQA(Component):
    def __init__(self):
        super().__init__()
        model_kwargs = {"model": "internlm2"}
        self.generator = Generator(
            model_client=OllamaClient(),
            model_kwargs=model_kwargs,
        )

    def call(self, input: dict) -> str:
        return self.generator.call({"input_str": str(input)})


if __name__ == "__main__":
    qa = SimpleQA()
    print(qa("What is the capital of France?"))
