from dataclasses import dataclass, field

from lightrag.core import Component, Generator, DataClass

# fun_to_component, Sequential
from lightrag.components.model_client import GroqAPIClient
from lightrag.components.output_parsers import JsonOutputParser
from lightrag.utils import setup_env

setup_env()


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


@dataclass
class QAOutput(DataClass):
    explanation: str = field(
        metadata={"desc": "A brief explanation of the concept in one sentence."}
    )
    example: str = field(metadata={"desc": "An example of the concept in a sentence."})


# @fun_to_component
# def to_qa_output(data: dict) -> QAOutput:
#     return QAOutput.from_dict(data)


qa_template = r"""<SYS>
You are a helpful assistant.
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
</SYS>
User: {{input_str}}
You:"""


class QA(Component):
    def __init__(self):
        super().__init__()

        parser = JsonOutputParser(data_class=QAOutput, return_data_class=True)
        self.generator = Generator(
            model_client=GroqAPIClient(),
            model_kwargs={"model": "llama3-8b-8192"},
            template=qa_template,
            prompt_kwargs={"output_format_str": parser.format_instructions()},
            output_processors=parser,
        )

    def call(self, query: str):
        return self.generator.call({"input_str": query})

    async def acall(self, query: str):
        return await self.generator.acall({"input_str": query})


def minimum_generator():
    from lightrag.core import Generator
    from lightrag.components.model_client import GroqAPIClient

    generator = Generator(
        model_client=GroqAPIClient(),
        model_kwargs={"model": "llama3-8b-8192"},
    )
    print(generator)
    prompt_kwargs = {"input_str": "What is LLM? Explain in one sentence."}
    generator.print_prompt(**prompt_kwargs)
    output = generator(
        prompt_kwargs=prompt_kwargs,
    )
    print(output)


def use_a_json_parser():
    from lightrag.core import Generator
    from lightrag.core.types import GeneratorOutput
    from lightrag.components.model_client import OpenAIClient
    from lightrag.core.string_parser import JsonParser

    output_format_str = """Your output should be formatted as a standard JSON object with two keys:
    {
        "explaination": "A brief explaination of the concept in one sentence.",
        "example": "An example of the concept in a sentence."
    }
    """

    generator = Generator(
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-3.5-turbo"},
        prompt_kwargs={"output_format_str": output_format_str},
        output_processors=JsonParser(),
    )

    prompt_kwargs = {"input_str": "What is LLM?"}
    generator.print_prompt(**prompt_kwargs)

    output: GeneratorOutput = generator(prompt_kwargs=prompt_kwargs)
    print(output)
    print(type(output.data))
    print(output.data)


def use_its_own_template():
    from lightrag.core import Generator
    from lightrag.components.model_client import GroqAPIClient

    template = r"""<SYS>{{task_desc_str}}</SYS>
User: {{input_str}}
You:"""
    generator = Generator(
        model_client=GroqAPIClient(),
        model_kwargs={"model": "llama3-8b-8192"},
        template=template,
        prompt_kwargs={"task_desc_str": "You are a helpful assistant"},
    )

    prompt_kwargs = {"input_str": "What is LLM?"}

    generator.print_prompt(
        **prompt_kwargs,
    )
    output = generator(
        prompt_kwargs=prompt_kwargs,
    )
    print(output)


def use_model_client_enum_to_switch_client():
    from lightrag.core import Generator
    from lightrag.core.types import ModelClientType

    generator = Generator(
        model_client=ModelClientType.OPENAI(),  # or ModelClientType.GROQ()
        model_kwargs={"model": "gpt-3.5-turbo"},
    )
    print(generator)
    prompt_kwargs = {"input_str": "What is LLM? Explain in one sentence."}
    generator.print_prompt(**prompt_kwargs)
    output = generator(
        prompt_kwargs=prompt_kwargs,
    )
    print(output)


def create_purely_from_config():

    from lightrag.utils.config import new_component
    from lightrag.core import Generator

    config = {
        "generator": {
            "component_name": "Generator",
            "component_config": {
                "model_client": {
                    "component_name": "GroqAPIClient",
                    "component_config": {},
                },
                "model_kwargs": {
                    "model": "llama3-8b-8192",
                },
            },
        }
    }

    generator: Generator = new_component(config["generator"])
    print(generator)

    prompt_kwargs = {"input_str": "What is LLM? Explain in one sentence."}
    generator.print_prompt(**prompt_kwargs)
    output = generator(
        prompt_kwargs=prompt_kwargs,
    )
    print(output)


def create_purely_from_config_2():

    from lightrag.core import Generator

    config = {
        "model_client": {
            "component_name": "GroqAPIClient",
            "component_config": {},
        },
        "model_kwargs": {
            "model": "llama3-8b-8192",
        },
    }

    generator: Generator = Generator.from_config(config)
    print(generator)

    prompt_kwargs = {"input_str": "What is LLM? Explain in one sentence."}
    generator.print_prompt(**prompt_kwargs)
    output = generator(
        prompt_kwargs=prompt_kwargs,
    )
    print(output)


if __name__ == "__main__":
    qa1 = SimpleQA()
    answer = qa1("What is LightRAG?")
    print(qa1)

    qa2 = QA()
    answer = qa2("What is LLM?")
    print(qa2)
    print(answer)
    qa2.generator.print_prompt(
        output_format_str=qa2.generator.output_processors.format_instructions(),
        input_str="What is LLM?",
    )

    minimum_generator()
    # use_a_json_parser()
    # use_its_own_template()
    # use_model_client_enum_to_switch_client()
    # create_purely_from_config()
    # create_purely_from_config_2()
