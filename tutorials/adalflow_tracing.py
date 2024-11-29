"""
This script demonstrates the usage of AdalFlow's tracing functionality.
It shows how to track Generator states and changes during development.
"""

import os
from getpass import getpass
from adalflow.tracing import trace_generator_states
from adalflow.core import Generator
import adalflow as adal
from adalflow.components.model_client import OpenAIClient


def setup_environment():
    """Setup API keys and environment variables."""
    # In a production environment, you might want to use environment variables
    # or a configuration file instead of getpass
    if "OPENAI_API_KEY" not in os.environ:
        openai_api_key = getpass("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = openai_api_key

    if "GROQ_API_KEY" not in os.environ:
        groq_api_key = getpass("Please enter your GROQ API key: ")
        os.environ["GROQ_API_KEY"] = groq_api_key

    print("API keys have been set.")


# Define the template for the doctor QA system
template_doc = r"""<SYS> You are a doctor </SYS> User: {{input_str}}"""


@trace_generator_states()
class DocQA(adal.Component):
    """
    A component that uses a Generator to answer medical questions.
    The @trace_generator_states decorator automatically tracks changes
    to any Generator attributes in this class.
    """

    def __init__(self):
        super(DocQA, self).__init__()
        self.generator = Generator(
            template=template_doc,
            model_client=OpenAIClient(),
            model_kwargs={"model": "gpt-4-turbo-preview"},
        )

    def call(self, query: str) -> str:
        """
        Process a medical query and return the response.

        Args:
            query: The medical question to be answered

        Returns:
            The generated response from the doctor AI
        """
        return self.generator(prompt_kwargs={"input_str": query}).data


def main():
    """Main function to demonstrate tracing functionality."""
    # Setup environment
    setup_environment()

    # Initialize the DocQA component
    doc_qa = DocQA()

    # Example queries
    queries = [
        "What are the common symptoms of the flu?",
        "How can I manage my allergies?",
        "What should I do for a minor burn?",
    ]

    # Process each query
    for query in queries:
        print(f"\nQuery: {query}")
        response = doc_qa.call(query)
        print(f"Response: {response}")

    print("\nNote: Generator states have been logged to the traces directory.")
    print("You can find the logs in: ./traces/DocQA/generator_state_trace.json")


if __name__ == "__main__":
    main()
