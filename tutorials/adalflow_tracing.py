"""
This script demonstrates the usage of AdalFlow's tracing functionality.
It shows how to track Generator states and changes during development.
"""

from adalflow.tracing import trace_generator_states
from adalflow.core import Generator
import adalflow as adal
from adalflow.components.model_client import OpenAIClient


template_doc = r"""<SYS> You are a doctor </SYS> User: {{input_str}}"""


@trace_generator_states()
class DocQA(adal.Component):

    def __init__(self):
        super(DocQA, self).__init__()
        self.generator = Generator(
            template=template_doc,
            model_client=OpenAIClient(),
            model_kwargs={"model": "gpt-4-turbo-preview"},
        )

    def call(self, query: str) -> str:

        return self.generator(prompt_kwargs={"input_str": query}).data


def main():

    doc_qa = DocQA()

    queries = [
        "What are the common symptoms of the flu?",
        "How can I manage my allergies?",
        "What should I do for a minor burn?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        response = doc_qa.call(query)
        print(f"Response: {response}")

    print("\nNote: Generator states have been logged to the traces directory.")
    print("You can find the logs in: ./traces/DocQA/generator_state_trace.json")


if __name__ == "__main__":
    from adalflow.utils import setup_env

    setup_env()
    main()
