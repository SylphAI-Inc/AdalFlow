from core.generator import Generator
from core.component import Component

from components.api_client import GroqAPIClient

import utils.setup_env


class SimpleQA(Component):
    def __init__(self):
        super().__init__()
        self.generator = Generator(
            model_client=GroqAPIClient,  # other options OpenAIClient, AnthropicClient, GoogleGenAIClient,
            model_kwargs={"model": "llama3-8b-8192"},
            preset_prompt_kwargs={
                "task_desc_str": "You are a helpful assistant and with a great sense of humor."
            },
        )

    def call(self, query: str) -> str:
        return self.generator.call(input=query)


if __name__ == "__main__":
    simple_qa = SimpleQA()
    print(simple_qa)
    print("show the system prompt")
    simple_qa.generator.print_prompt()
    print("Answer:")
    print(simple_qa.call("What is the capital of France?"))
