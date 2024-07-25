# This code implements a simple Question-Answering system using the LightRAG framework.
# It uses a language model to generate answers to user queries.

from lightrag.core.generator import Generator
from lightrag.core.component import Component
from lightrag.components.model_client import LiteClient
from lightrag.utils import setup_env

# Setup environment variables (e.g., API keys). Remove this in production.
setup_env("C:/Users/jean\Documents/molo/LightRAG/.env")


class SimpleQA(Component):
    def __init__(self):
        super().__init__()
        # Initialize the generator with a LiteClient using the deepseek-chat model
        # Other client options include OpenAIClient, AnthropicClient, GoogleGenAIClient
        self.generator = Generator(
            model_client=LiteClient(
                model="together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
            ),
            model_kwargs={
                "model": "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
            },  # Define which parameters are trainable
        )

    def call(self, query: str) -> str:
        # Generate an answer for the given query
        return self.generator.call(prompt_kwargs={"input_str": query})


if __name__ == "__main__":
    # Create an instance of SimpleQA
    simple_qa = SimpleQA()
    print(simple_qa)

    # Get the state dictionary (parameters) of the SimpleQA instance
    states = simple_qa.state_dict()
    print(f"states: {states}")

    # Display the system prompt used by the generator
    print("show the system prompt")
    simple_qa.generator.print_prompt()

    # Generate and print an answer to a sample question
    print("Answer:")
    print(simple_qa.call("What is the capital of France?"))
