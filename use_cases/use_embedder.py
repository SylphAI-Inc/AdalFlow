import asyncio
from typing import List, Any

from lightrag.core.embedder import Embedder
from lightrag.core.component import Component
from lightrag.components.model_client import OpenAIClient


class SimpleEmbedder(Component):
    """
    The embedder takes a list of queries leverage the api's batch processing capabilities
    """

    def __init__(self):
        super().__init__()
        model_kwargs = {
            "model": "text-embedding-3-small",
            "dimensions": 256,
            "encoding_format": "float",
        }
        self.embedder = Embedder(model_client=OpenAIClient(), model_kwargs=model_kwargs)

    def call(self, queries: List[str]) -> Any:
        return self.embedder(input=queries)

    async def acall(self, queries: List[str]) -> Any:
        return await self.embedder.acall(input=queries)


if __name__ == "__main__":
    import time

    queries = ["What is the capital of France?"] * 10

    embedder = SimpleEmbedder()
    print(embedder)
    t0 = time.time()
    print(embedder.call(queries))
    t1 = time.time()
    print(f"Total time for 1 sync call: {t1 - t0} seconds")

    async def make_async_call(query: str) -> str:
        return await embedder.acall(query)

    async def main():
        async_queries = [queries] * 10
        tasks = [make_async_call(query) for query in async_queries]

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        print(results)
        end_time = time.time()

        print(f"Total time for 10 async calls: {end_time - start_time} seconds")

    # Execute the asynchronous function
    asyncio.run(main())
