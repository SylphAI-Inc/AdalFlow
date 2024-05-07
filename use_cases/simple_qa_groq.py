"""
We just need to very basic generator that can be used to generate text from a prompt.
"""

from core.generator import Generator
from components.api_client.groq_client import GroqAPIClient

from core.component import Component

# TODO: make the environment variable loading more robust, and let users specify the .env path
import dotenv

dotenv.load_dotenv()


class SimpleQA(Component):
    def __init__(self):
        super().__init__()
        self.generator = Generator(
            model_client=GroqAPIClient(), model_kwargs={"model": "llama3-8b-8192"}
        )
        self.generator.print_prompt()

    def call(self, query: str) -> str:
        return self.generator.call(input=query)


if __name__ == "__main__":
    simple_qa = SimpleQA()
    print(simple_qa)
    print(simple_qa.call("What is the capital of France?"))
