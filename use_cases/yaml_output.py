from lightrag.core.component import Component
from lightrag.core.generator import Generator
from lightrag.components.model_client import GroqAPIClient
from lightrag.components.output_parsers import YamlOutputParser
from dataclasses import dataclass
from lightrag.utils import setup_env


from lightrag.core.base_data_class import DataClass, field
from lightrag.core.types import GeneratorOutput


@dataclass
class JokeOutput(DataClass):
    setup: str = field(metadata={"desc": "question to set up a joke"}, default="")
    punchline: str = field(metadata={"desc": "answer to resolve the joke"}, default="")


joke_example = JokeOutput(
    setup="Why did the scarecrow win an award?",
    punchline="Because he was outstanding in his field.",
)

setup_env()


# Our parser not only helps format the output format, but also examples
class JokeGenerator(Component):
    def __init__(self):
        super().__init__()
        yaml_parser = YamlOutputParser(data_class=JokeOutput, examples=[joke_example])
        self.generator = Generator(
            model_client=GroqAPIClient(),
            model_kwargs={"model": "llama3-8b-8192", "temperature": 1.0},
            prompt_kwargs={"output_format_str": yaml_parser.format_instructions()},
            output_processors=yaml_parser,
        )

    def call(self, query: str, model_kwargs: dict = {}) -> JokeOutput:
        response: GeneratorOutput = self.generator.call(
            prompt_kwargs={"input_str": query}, model_kwargs=model_kwargs
        )
        if response.error is None:
            output = JokeOutput.from_dict(response.data)
            return output
        else:
            None


if __name__ == "__main__":
    joke_generator = JokeGenerator()
    print(joke_generator)
    print("show the system prompt")
    joke_generator.generator.print_prompt()
    print("Answer:")
    answer = joke_generator.call("Tell me two jokes.", model_kwargs={"temperature": 1})
    print(answer)
    print(f"typeof answer: {type(answer)}")

    from langchain_core.output_parsers import (
        JsonOutputParser as LangchainJsonOutputParser,
    )
    from langchain_core.pydantic_v1 import BaseModel, Field
    from typing import TypeVar, Generic

    T = TypeVar("T")
    # 1. we dont include default value in the schema as it is how the program wants to handle it and not the job of llm
    # 2. nested class is defined in definitions.
    # we directly include in that type, and we should remove the additional "type" and make that part a dict instead of a string.
    ours = {
        "type": "MyStepOutput",
        "properties": {
            "step": {"type": "int", "desc": "The order of the step in the agent"},
            "action": {
                "type": "{'type': 'FunctionExpression', 'properties': {'thought': {'type': 'Optional[str]', 'desc': 'Why the function is called'}, 'action': {'type': 'str', 'desc': 'FuncName(<kwargs>) Valid function call expression. Example: \"FuncName(a=1, b=2)\" Follow the data type specified in the function parameters.e.g. for Type object with x,y properties, use \"ObjectType(x=1, y=2)'}}, 'required': ['action']}",
                "desc": "The action the agent takes at this step",
            },
            "function": {
                "type": "Optional[{'type': 'Function', 'properties': {'thought': {'type': 'Optional[str]', 'desc': 'Why the function is called'}, 'name': {'type': 'str', 'desc': 'The name of the function'}, 'args': {'type': 'Optional[List[object]]', 'desc': 'The positional arguments of the function'}, 'kwargs': {'type': 'Optional[Dict[str, object]]', 'desc': 'The keyword arguments of the function'}}, 'required': []}]",
                "desc": "The parsed function from the action",
            },
            "observation": {
                "type": "Optional[str]",
                "desc": "The execution result shown for this action",
            },
        },
        "required": [],
    }
    langchains = {
        "properties": {
            "setup": {
                "title": "Setup",
                "description": "question to set up a joke",
                "default": "",
                "allOf": [{"$ref": "#/definitions/JokeOutput"}],
            },
            "punchline": {
                "title": "Punchline",
                "description": "answer to resolve the joke",
                "default": "",
                "type": "string",
            },
        },
        "definitions": {
            "JokeOutput": {
                "title": "JokeOutput",
                "type": "object",
                "properties": {
                    "setup": {
                        "title": "Setup",
                        "default": "",
                        "desc": "question to set up a joke",
                        "type": "string",
                    },
                    "punchline": {
                        "title": "Punchline",
                        "default": "",
                        "desc": "answer to resolve the joke",
                        "type": "string",
                    },
                },
            }
        },
    }

    # subtype: allOf": [{"$ref": "#/definitions/JokeOutput"}]},
    definitions = {
        "definitions": {
            "JokeOutput": {
                "title": "JokeOutput",
                "type": "object",
                "properties": {
                    "setup": {
                        "title": "Setup",
                        "default": "",
                        "desc": "question to set up a joke",
                        "type": "string",
                    },
                    "punchline": {
                        "title": "Punchline",
                        "default": "",
                        "desc": "answer to resolve the joke",
                        "type": "string",
                    },
                },
            }
        }
    }

    class JokeOutputV1(BaseModel, Generic[T]):
        setup: JokeOutput = Field(description="question to set up a joke", default="")
        punchline: str = Field(description="answer to resolve the joke", default="")

    json_parser = LangchainJsonOutputParser(pydantic_object=JokeOutputV1)
    print("langchain instruction")
    # failed to detect the generic type of the pydantic object
    print(json_parser.get_format_instructions())
