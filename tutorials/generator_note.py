from dataclasses import dataclass, field

from adalflow.core import Component, Generator, DataClass

from adalflow.components.model_client import GroqAPIClient
from adalflow.components.output_parsers import JsonOutputParser
from adalflow.utils import setup_env

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
    from adalflow.core import Generator
    from adalflow.components.model_client import GroqAPIClient

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
    from adalflow.core import Generator
    from adalflow.core.types import GeneratorOutput
    from adalflow.components.model_client import OpenAIClient
    from adalflow.core.string_parser import JsonParser

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
    from adalflow.core import Generator
    from adalflow.components.model_client import GroqAPIClient

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
    from adalflow.core import Generator
    from adalflow.core.types import ModelClientType

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

    from adalflow.utils.config import new_component
    from adalflow.core import Generator

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

    from adalflow.core import Generator

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


def simple_query():

    from adalflow.core import Generator
    from adalflow.components.model_client.openai_client import OpenAIClient

    gen = Generator(
        model_client=OpenAIClient(),
        model_kwargs={
            "model": "o3-mini",
        },
    )

    response = gen({"input_str": "What is LLM?"})
    print(response)


def customize_template():

    import adalflow as adal

    # the template has three variables: system_prompt, few_shot_demos, and input_str
    few_shot_template = r"""<START_OF_SYSTEM_PROMPT>
{{system_prompt}}
{# Few shot demos #}
{% if few_shot_demos is not none %}
Here are some examples:
{{few_shot_demos}}
{% endif %}
<END_OF_SYSTEM_PROMPT>
<START_OF_USER>
{{input_str}}
<END_OF_USER>"""

    object_counter = Generator(
        model_client=adal.GroqAPIClient(),
        model_kwargs={
            "model": "llama3-8b-8192",
        },
        template=few_shot_template,
        prompt_kwargs={
            "system_prompt": "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
        },
    )

    question = "I have a flute, a piano, a trombone, four stoves, a violin, an accordion, a clarinet, a drum, two lamps, and a trumpet. How many musical instruments do I have?"
    response = object_counter(prompt_kwargs={"input_str": question})
    print(response)

    object_counter.print_prompt(input_str=question)

    # use an int parser

    from adalflow.core.string_parser import IntParser

    object_counter = Generator(
        model_client=adal.GroqAPIClient(),
        model_kwargs={
            "model": "llama3-8b-8192",
        },
        template=few_shot_template,
        prompt_kwargs={
            "system_prompt": "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
        },
        output_processors=IntParser(),
    )

    response = object_counter(prompt_kwargs={"input_str": question})
    print(response)
    print(type(response.data))

    # use customize parser
    import re

    @adal.func_to_data_component
    def parse_integer_answer(answer: str):
        try:
            numbers = re.findall(r"\d+", answer)
            if numbers:
                answer = int(numbers[-1])
            else:
                answer = -1
        except ValueError:
            answer = -1

        return answer

    object_counter = Generator(
        model_client=adal.GroqAPIClient(),
        model_kwargs={
            "model": "llama3-8b-8192",
        },
        template=few_shot_template,
        prompt_kwargs={
            "system_prompt": "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value.",
        },
        output_processors=parse_integer_answer,
    )

    response = object_counter(prompt_kwargs={"input_str": question})
    print(response)
    print(type(response.data))

    template = r"""<START_OF_SYSTEM_PROMPT>
{{system_prompt}}
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
<END_OF_SYSTEM_PROMPT>
<START_OF_USER>
{{input_str}}
<END_OF_USER>"""

    from dataclasses import dataclass, field

    @dataclass
    class QAOutput(DataClass):
        thought: str = field(
            metadata={
                "desc": "Your thought process for the question to reach the answer."
            }
        )
        answer: int = field(metadata={"desc": "The answer to the question."})

        __output_fields__ = ["thought", "answer"]

    parser = adal.DataClassParser(
        data_class=QAOutput, return_data_class=True, format_type="json"
    )

    object_counter = Generator(
        model_client=adal.GroqAPIClient(),
        model_kwargs={
            "model": "llama3-8b-8192",
        },
        template=template,
        prompt_kwargs={
            "system_prompt": "You will answer a reasoning question. Think step by step. ",
            "output_format_str": parser.get_output_format_str(),
        },
        output_processors=parser,
    )

    response = object_counter(prompt_kwargs={"input_str": question})
    print(response)

    object_counter.print_prompt(input_str=question)


if __name__ == "__main__":
    qa1 = SimpleQA()
    answer = qa1("What is adalflow?")
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
    simple_query()
    customize_template()
    # use_a_json_parser()
    # use_its_own_template()
    # use_model_client_enum_to_switch_client()
    # create_purely_from_config()
    # create_purely_from_config_2()
