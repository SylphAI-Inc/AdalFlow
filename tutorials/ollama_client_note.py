from lightrag.core.generator import Generator
from lightrag.core.component import Component
from lightrag.core.embedder import Embedder
from lightrag.components.model_client import OllamaClient
from dotenv import load_dotenv
import asyncio
from lightrag.core.types import ModelType


# To get started with Ollama we're going to need to install it.
# Navigate to https://ollama.com/download in order to get ollama installed locally.
# Once you have ollama installed, we can pull the model we want to use with LightRAG.
## `ollama pull qwen2:0.5b` will allow you to download the Qwen2 model ~ 352MB
## `ollama pull jina/jina-embeddings-v2-small-en` will allow you to download the Jina Embedding model ~ 66MB
## **After installation, ollama should already be running in the background.**
# See https://ollama.com/library/qwen2:0.5b
# See https://ollama.com/jina/jina-embeddings-v2-small-en
# Once you've completed the steps above, you're ready to run the demo.

# We'll start with a poem and then embed the topics to show the usage of the asyncronous client.
# Create the template for the Generator and define the model to use
# Note: You can pass the host to OllamaClient(host='remotehost:11434') if you're running ollama on another server.
#       DEFAULT is http://localhost:11434 - You'll need to ensure that you have ollama running on this port
#       See: https://github.com/ollama/ollama/blob/main/docs/linux.md for linux system service instructions. (if you need to restart the service)

load_dotenv()


class PoemGenerator(Component):
    def __init__(self) -> None:
        super().__init__()
        self.generator = Generator(
            model_client=OllamaClient(host="http://localhost:11434"),
            model_kwargs={"model": "qwen2:0.5b"},
            template="""Generate a haiku about {{topic}}.""",
        )
        self.embedder = Embedder(
            model_client=OllamaClient(host="http://localhost:11434"),
            model_kwargs={"model": "jina/jina-embeddings-v2-base-en:latest"},
        )

    def call(self, prompt_kwargs, model_type: ModelType):
        if model_type == ModelType.LLM:
            return self.generator.call(prompt_kwargs)
        if model_type == ModelType.EMBEDDER:
            return self.embedder.call(prompt_kwargs["topic"])

    async def acall(self, topics: list[str], model_type: ModelType) -> list[str]:
        results = []
        if model_type == ModelType.LLM:
            for topic in topics:
                result = await self.generator.acall({"topic": topic})
                results.append(result.data)
            return results
        if model_type == ModelType.EMBEDDER:
            for topic in topics:
                result = await self.embedder.acall(topic)
                results.append(result.data)
            return results


if __name__ == "__main__":
    # Create the kwargs to call the model with. This is going to simply use the topic our prompt expects.
    prompt_kwargs = {"topic": "the Utah sky, in the winter, after a midnight storm"}
    topics = ["summer in Belize", "fall in Morocco", "spring in Saigon"]
    poem_gen = PoemGenerator()
    # Sync Generator and Embedding Output
    sync_output = poem_gen.call(prompt_kwargs, ModelType.LLM)
    sync_embed = poem_gen.call(prompt_kwargs, ModelType.EMBEDDER)
    print("Sync Poem:")
    print(sync_output.data)
    print("Sync Embedding:")
    print(sync_embed.data)

    # Async Generator and Embedding Output
    async_output = asyncio.run(poem_gen.acall(topics, ModelType.LLM))
    async_embed = asyncio.run(poem_gen.acall(topics, ModelType.EMBEDDER))

    # Print the output
    print("Multiple Poems:\n")
    for idx, poem in enumerate(async_output):
        print(f"Topic: {topics[idx]}")
        print(f"{poem}\n")

    print("\nEmbedded Topics:\n")
    for idx, poem in enumerate(async_embed):
        print(f"Topic: {topics[idx]}")
        print(f"{poem}\n")
