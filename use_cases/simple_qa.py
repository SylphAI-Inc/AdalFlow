import asyncio
from core.generator import Generator
from core.openai_client import OpenAIClient

from core.component import Component

import utils.setup_env


class SimpleQA(Component):
    def __init__(self):
        super().__init__()
        model_kwargs = {"model": "gpt-3.5-turbo"}
        self.generator = Generator[int](
            model_client=OpenAIClient(), model_kwargs=model_kwargs
        )
        self.generator.print_prompt()

    def call(self, query: str) -> str:
        return self.generator(input=query)

    async def acall(self, query: str) -> str:
        return await self.generator.acall(input=query)


if __name__ == "__main__":
    import time

    simple_qa = SimpleQA()
    print(simple_qa)
    t0 = time.time()
    print(simple_qa.call("What is the capital of France?"))
    t1 = time.time()
    print(f"Total time for 1 sync call: {t1 - t0} seconds")

    async def make_async_call(query: str) -> str:
        return await simple_qa.acall(query)

    async def main():
        queries = ["What is the capital of France?"] * 10
        tasks = [make_async_call(query) for query in queries]

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        print(f"Total time for 10 async calls: {end_time - start_time} seconds")

    # Execute the asynchronous function
    asyncio.run(main())
