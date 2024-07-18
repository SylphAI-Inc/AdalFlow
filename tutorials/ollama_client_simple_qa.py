from lightrag.core.generator import Generator
from lightrag.core.component import Component
from lightrag.core.embedder import Embedder
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

kwargs = {
    "model": "jina/jina-embeddings-v2-base-en:latest",
}

response = client.embeddings(
    model="jina/jina-embeddings-v2-base-en:latest",
    prompt="Welcome",
)
print(response)


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


def test_embedder():
    # ollama pull jina/jina-embeddings-v2-base-en:latest

    model_kwargs = {"model": "jina/jina-embeddings-v2-base-en:latest"}
    embedder = Embedder(model_client=OllamaClient(), model_kwargs=model_kwargs)
    response = embedder.call(input="Hello world")
    print(response)


if __name__ == "__main__":
    test_embedder()
# qa = SimpleQA()
# print(qa("What is the capital of France?"))
