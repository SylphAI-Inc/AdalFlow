from lightrag.core.generator import Generator
from lightrag.core.component import Component

from lightrag.components.model_client import GroqAPIClient

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
            trainable_params=[
                "task_desc_str"
            ],  # 1 we need to clearly define which is trainable.
        )

    def init_parameters(self):
        self.generator.task_desc_str.update_value(
            "You are a helpful assistant and with a great sense of humor."
        )

    def call(self, query: str) -> str:
        return self.generator.call(input=query)


if __name__ == "__main__":
    # TODO: convert some of this code to pytest for states
    simple_qa = SimpleQA()
    print(simple_qa)
    simple_qa.init_parameters()
    states = simple_qa.state_dict()
    print(f"states: {states}")  # conv1.weight, conv1.bias, fc1.weight, fc1.bias

    simple_qa_2 = SimpleQA()
    states_before = simple_qa_2.state_dict()
    print(f"states_before: {states_before}")
    simple_qa_2.load_state_dict(states)
    states_2 = simple_qa_2.state_dict()
    print(f"states_2: {states_2}")

    print("show the system prompt")
    simple_qa.generator.print_prompt()
    print("Answer:")
    print(simple_qa.call("What is the capital of France?"))
