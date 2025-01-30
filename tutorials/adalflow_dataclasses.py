from dataclasses import dataclass, field
from typing import Dict
import adalflow as adal
from adalflow.components.model_client import GroqAPIClient

# Define the QA template using jinja2 syntax
qa_template = r"""<SYS>
You are a helpful assistant.
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
</SYS>
<USER> {{input_str}} </USER>"""


# Define the output structure using dataclass
@dataclass
class BasicQAOutput(adal.DataClass):
    explanation: str = field(
        metadata={"desc": "A brief explanation of the concept in one sentence."}
    )
    example: str = field(metadata={"desc": "An example of the concept in a sentence."})
    __output_fields__ = ["explanation", "example"]


# Define the QA component
class QA(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()
        parser = adal.DataClassParser(data_class=BasicQAOutput, return_data_class=True)
        self.generator = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=qa_template,
            prompt_kwargs={"output_format_str": parser.get_output_format_str()},
            output_processors=parser,
        )

    def call(self, query: str):
        """Synchronous call to generate response"""
        return self.generator.call({"input_str": query})

    async def acall(self, query: str):
        """Asynchronous call to generate response"""
        return await self.generator.acall({"input_str": query})


def run_basic_example():
    """Run a basic example of the QA component"""
    qa = QA(
        model_client=GroqAPIClient(),
        model_kwargs={"model": "llama3-8b-8192"},
    )
    response = qa("What is LLM?")
    print("\nResponse:")
    print(response)
    print(f"BasicQAOutput: {response.data}")
    print(f"Explanation: {response.data.explanation}")
    print(f"Example: {response.data.example}")


if __name__ == "__main__":
    from adalflow.utils import setup_env

    setup_env()
    print("Running basic QA example...")
    run_basic_example()
