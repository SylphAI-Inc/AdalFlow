from lightrag.components.output_parsers import JsonOutputParser
from lightrag.core.generator import Generator
from lightrag.core.component import Component
from lightrag.components.model_client import OllamaClient
from lightrag.core.tool_manager import ToolManager
from lightrag.core.types import Function, GeneratorOutput
from lightrag.core.container import Sequential
import boto3


# This file will contain simple generator examples, a fun poem, as well as an example of local llms ability to call functions, we'll use the AWS S3 API.

# Let's say we want to format the objects in our S3 buckets, without sending our information to a proprietary LLM API, in a particular fashion...


# Create the functions utilizing the boto3 library.
def get_s3_buckets() -> dict:
    client = boto3.client("s3")
    return client.list_buckets()


def list_objects(bucket: str):
    client = boto3.client("s3")
    return client.list_objects_v2(Bucket=bucket)


# Create components that will serve as function calls to our local LLM
class S3Tools(Component):
    def __init__(
        self, model_client, model_kwargs, prompt_kwargs, output_processor, manager
    ):
        super().__init__()
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs=prompt_kwargs,
            output_processors=output_processor,
        )
        self.manager = manager

    # Override the call function to return the output of the function call.
    def call(self, input: dict) -> str:
        output: GeneratorOutput = self.generator(input)
        func = Function.from_dict(output.data)
        func_output = self.manager.execute_func(func)
        return func_output.output


# Create a component to synthesize the response.
class Synth(Component):
    def __init__(self, model_client, model_kwargs, prompt_kwargs):
        super().__init__()
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs=prompt_kwargs,
        )

    def call(self, input: dict) -> str:
        return self.generator.call({"input_str": str(input)})


if __name__ == "__main__":
    model_kwargs = {"model": "internlm2"}
    # Let's start with something simple, generate a poem
    gen = Generator(model_client=OllamaClient(), model_kwargs=model_kwargs)
    output = gen(
        {
            "input_str": "Generate a poem about the Utah sky, in the winter, after a midnight storm."
        }
    )
    print(output.data)

    # Let's spice it up a bit!
    # Add functions to array and define ToolManager
    functions = [get_s3_buckets, list_objects]
    manager = ToolManager(tools=functions)
    func_parser = JsonOutputParser(
        data_class=Function, exclude_fields=["thought", "args"]
    )

    # Define the model and prompt kwargs
    prompt_kwargs = {
        "tools_str": manager.yaml_definitions,
        "output_format_str": func_parser.format_instructions(),
    }
    # Define the input into the components
    input = {
        "input_str": "List out the objects in the bucket named bpdata-clean-jobdescription-dev. Only output the names."
    }

    # Define the sequnce to read an S3 bucket. We're going to pass a context_str into the Synth component, which is the initial input string
    seq = Sequential(
        S3Tools(
            model_client=OllamaClient(),
            model_kwargs=model_kwargs,
            prompt_kwargs=prompt_kwargs,
            output_processor=func_parser,
            manager=manager,
        ),
        Synth(
            model_client=OllamaClient(),
            model_kwargs=model_kwargs,
            prompt_kwargs={"context_str": input["input_str"]},
        ),
    )
    # Run the sequence
    result = seq(input)
    print(result.data)
