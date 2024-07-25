# This is a tutorial on how to use the litellm client, let's build a simple linkedin post generator.
from lightrag.core.generator import Generator, Component
from lightrag.components.model_client import LiteClient
from lightrag.core.embedder import Embedder
from lightrag.core.types import ModelType
import asyncio

# To get started with litellm as provider, we need to provide the model name,set the correct api key and the prompt
# This is a short explanation of how provide the model name : https://docs.litellm.ai/docs/#basic-usage
# For most popular providers like : OpenAI, anthropic, you can pass directly the model name as a string
# LiteClient(model="claude-3-5-sonnet-20240620")
# LiteClient(model="gpt-4o")
# For other you need to provide firstly the provider name and then the model name
# LiteClient(model="deepseek/deepseek-chat"), discover more here : https://docs.litellm.ai/docs/providers

#we will start with linkedin post generator and then we will generate a post

from lightrag.utils import setup_env # ensure you have .env with OPENAI_API_KEY
setup_env("C:/Users/jean\Documents/molo/LightRAG/.env")  # need to setup env, remove it in production or let's empty


paragraph="""
Prior to GPT-4o, you could use Voice Mode to talk to ChatGPT with latencies of 2.8 seconds (GPT-3.5) and 5.4 seconds (GPT-4) on average. To achieve this, Voice Mode is a pipeline of three separate models: one simple model transcribes audio to text, GPT-3.5 or GPT-4 takes in text and outputs text, and a third simple model converts that text back to audio. This process means that the main source of intelligence, GPT-4, loses a lot of information—it can’t directly observe tone, multiple speakers, or background noises, and it can’t output laughter, singing, or express emotion.

With GPT-4o, we trained a single new model end-to-end across text, vision, and audio, meaning that all inputs and outputs are processed by the same neural network. Because GPT-4o is our first model combining all of these modalities, we are still just scratching the surface of exploring what the model can do and its limitations.

Explorations of capabilities.
"""


class LinkedinGenerator(Component):
    def __init__(self)-> None:
        super().__init__()
        self.generator = Generator(
            model_client=LiteClient(model="together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
            model_kwargs={"model": "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"},
            template="""Generate a linkedin post about {{news}}. Be accurate, use bullets points and emojis to make it more engaging.""",
        )
        self.embedder = Embedder(
            model_client=LiteClient(model="together_ai/togethercomputer/m2-bert-80M-2k-retrieval"),
              model_kwargs={"model": "huggingface/microsoft/codebert-base"} 
        )
    def call(self, prompt_kwargs, model_type: ModelType):
        if model_type == ModelType.LLM:
            return self.generator.call(prompt_kwargs)
        if model_type == ModelType.EMBEDDER:
            return self.embedder.call(prompt_kwargs["news"])
        
    async def acall(self, news: list[str], model_type: ModelType) -> list[str]:
        results = []
        if model_type == ModelType.LLM:
            for new in news:
                result = await self.generator.acall({"news": new})
                results.append(result.data)
            return results
        if model_type == ModelType.EMBEDDER:
            for new in news:
                result = await self.embedder.acall(new)
                results.append(result.data)
            return results
        
        
        
if __name__ == "__main__":
    prompt_kwargs = {"news": paragraph}
    generator=LinkedinGenerator()
    # Sync Generator 
    sync_output = generator.call(prompt_kwargs, ModelType.LLM)
    print("Sync Poem:")
    print(sync_output.data)
    
    #and Embedding Output
    sync_embed = generator.call(prompt_kwargs, ModelType.EMBEDDER)
    print("Sync Embedding:")
    print(sync_embed) # this output desn't have data attribute
    
    #async function completed
    news = ["discovers america", "fall of berlin wall"]
    async_output = asyncio.run(generator.acall(news, ModelType.LLM))
    for idx, post in enumerate(async_output):
        print(f"Post {idx+1}: {post}")
        
   #async embedding
    news = ["discovers america", "fall of berlin wall"]
    async_embed = asyncio.run(generator.acall(news, ModelType.EMBEDDER))
    for idx, post in enumerate(async_embed):
        print(f"Embeddings {idx+1}: {post}")
        
# commment each step to get the results, news is used maby times in the code 