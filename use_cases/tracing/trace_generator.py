from core.generator import Generator
from core.component import Component

from components.api_client import GroqAPIClient
from tracing.decorators import trace_generator

import utils.setup_env


@trace_generator()
class SimpleQA(Component):
    def __init__(self):
        super().__init__()
        self.generator = Generator(
            model_client=GroqAPIClient(),
            model_kwargs={"model": "llama3-8b-8192"},
            preset_prompt_kwargs={
                "task_desc_str": "You are a helpful assistant and with a great sense of humor. Second edition.",
            },
            trainable_params=[
                "task_desc_str"
            ],  # 1 we need to clearly define which is trainable.
        )

    def call(self, query: str) -> str:
        return self.generator.call(prompt_kwargs={"input_str": query})


if __name__ == "__main__":
    simple_qa = SimpleQA()
    print(simple_qa)
    states = simple_qa.state_dict()  # only prameters
    print(f"states: {states}")

    print("show the system prompt")
    simple_qa.generator.print_prompt()
    print("Answer:")
    print(simple_qa.call("What is the capital of France?"))
