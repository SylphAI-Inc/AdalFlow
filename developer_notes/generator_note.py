from lightrag.core import Component, Generator
from lightrag.components.model_client import GroqAPIClient
from lightrag.utils import setup_env  # noqa


class SimpleQA(Component):
    def __init__(self):
        super().__init__()
        template = r"""<SYS>
        You are a helpful assistant.
        </SYS>
        User: {{input_str}}
        You:
        """
        self.generator = Generator(
            model_client=GroqAPIClient(),
            model_kwargs={"model": "llama3-8b-8192"},
            template=template,
        )

    def call(self, query):
        return self.generator({"input_str": query})

    async def acall(self, query):
        return await self.generator.acall({"input_str": query})


qa = SimpleQA()
answer = qa("What is LightRAG?")
print(qa)
