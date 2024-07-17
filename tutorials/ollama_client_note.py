from lightrag.core.generator import Generator
from lightrag.components.model_client import OllamaClient
import asyncio


# To get started with Ollama we're going to need to install it.
# Navigate to https://ollama.com/download in order to get ollama installed locally.
# Once you have ollama installed, we can pull the model we want to use with LightRAG.
## `ollama pull qwen2:0.5b` will allow you to download the Qwen2 model ~ 352MB
## **After installation, ollama should already be running in the background.**
# See https://ollama.com/library/qwen2:0.5b
# Once you've completed the steps above, you're ready to run the demo.

# We'll define an async function ahead of time to illustrate how to use the asyncronous client
async def poems_async(poem_generator: Generator, topics: list[str]) -> list[str]:
    results = []
    for topic in topics:
        result = await poem_generator.acall({"topic": topic})
        results.append(result.data)
    return results

# Let's show the clients in action
if __name__ == "__main__":
    # We'll start with a poem
    # Create the template for the Generator and define the model to use
    poem_prompt_template = """Generate a haiku about {{topic}}."""
    model_kwargs = {"model": "qwen2:0.5b"}
    poem_generator = Generator(
        model_client=OllamaClient(),
        model_kwargs=model_kwargs,
        template=poem_prompt_template,
    )
    # Create the kwargs to call the model with. This is going to simply use the topic our prompt expects.
    prompt_kwargs = {"topic": "the Utah sky, in the winter, after a midnight storm"}
    output = poem_generator(prompt_kwargs)
    print(f"Single Poem:\n{output.data}\n")
    

    # Now, let's show how we can get asynchronous output
    topics = ["summer in Belize", "fall in Morocco", "spring in Saigon"]
    poems = asyncio.run(poems_async(poem_generator, topics))
    print(f"Multiple Poems:\n")
    for idx, poem in enumerate(poems):
        print(f"Topic: {topics[idx]}")
        print(f"{poem}\n")

