"""
Tool is LLM's extended capability which is one of the core design pattern of Agent. All tools can be wrapped in a FunctionTool class.
This helps to standardize the tool interface and metadata to communicate with the Agent.
"""

from typing import Any, Optional, Dict, Callable, Awaitable, Union
from inspect import iscoroutinefunction

from dataclasses import dataclass, field

import json

from lightrag.core.types import FunctionDefinition, ToolOutput, Function
from lightrag.core import Component
from lightrag.core.functional import get_fun_schema

AsyncCallable = Callable[..., Awaitable[Any]]


class FunctionTool(Component):
    __doc__ = r"""

    Handles:
    * Create metadata by calling get_fun_schema.
    * Parse the output of LLM.
    * execute the function.

    There is almost no need to customize a FunctionTool, but you can do so if you want to.
    Support both positional and keyword arguments.
    NOTE:
    - at least one of fn or async_fn must be provided.
    - When both are provided, sync (call) will be used in __call__.
    """

    # metadata: ToolMetadata = field(
    #     default_factory=ToolMetadata, metadata={"desc": "Tool metadata"}
    # )
    # fn: Optional[Callable[..., Any]] = field(
    #     default=None, metadata={"desc": "Synchronous function"}
    # )
    # async_fn: Optional[AsyncCallable] = field(
    #     default=None, metadata={"desc": "Asynchronous function"}
    # )

    def __init__(
        self,
        fn: Union[Callable[..., Any], AsyncCallable],
        definition: Optional[FunctionDefinition] = None,
    ):
        super().__init__()
        assert fn is not None, "fn must be provided"

        self.fn = fn
        self._is_async = iscoroutinefunction(fn)
        print(f"self._is_async: {self._is_async}")

        self.definition = definition or self._create_fn_definition()

    def _create_fn_definition(self) -> FunctionDefinition:
        name = self.fn.__name__
        docstring = self.fn.__doc__
        description = f"{docstring}"
        # description = f"{name}{signature(self.fn)}\n{docstring}"
        fn_parameters = get_fun_schema(name, self.fn)
        return FunctionDefinition(
            func_name=name, func_desc=description, func_parameters=fn_parameters
        )

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        if self._is_async:
            raise ValueError("FunctionTool is asynchronous, use acall instead")
        tool_output = self.fn(*args, **kwargs)
        return ToolOutput(
            tool_name=self.metadata.name,
            raw_input={"args": args, "kwargs": kwargs},
            raw_output=tool_output,
        )

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        if not self._is_async:
            raise ValueError("FunctionTool is not asynchronous, use call instead")
        tool_output = await self.fn(*args, **kwargs)
        return ToolOutput(
            tool_name=self.metadata.name,
            raw_input={"args": args, "kwargs": kwargs},
            raw_output=tool_output,
        )


if __name__ == "__main__":
    from lightrag.utils import setup_env  # noqa
    from lightrag.utils import enable_library_logging

    enable_library_logging()

    def get_current_weather_1(location: str, unit: str = "fahrenheit"):
        """Get the current weather in a given location"""
        if "tokyo" in location.lower():
            return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
        elif "san francisco" in location.lower():
            return json.dumps(
                {"location": "San Francisco", "temperature": "72", "unit": unit}
            )
        elif "paris" in location.lower():
            return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
        else:
            return json.dumps({"location": location, "temperature": "unknown"})

    from lightrag.core.base_data_class import DataClassFormatType
    from dataclasses import dataclass, field

    @dataclass
    class Weather:
        location: str = field(
            metadata={"desc": "The city and state, e.g. San Francisco, CA"}
        )
        unit: str = field(metadata={"enum": ["celsius", "fahrenheit"]})

    def get_current_weather(weather: Union[Weather, Dict[str, Dict]]):
        """Get the current weather in a given location"""
        # LLM will work better with str and dict inputs.
        if isinstance(weather, str):
            weather = json.loads(weather)
        if isinstance(weather, dict):
            weather = Weather(**weather)

        if "tokyo" in weather.location.lower():
            return json.dumps(
                {"location": "Tokyo", "temperature": "10", "unit": weather.unit}
            )
        elif "san francisco" in weather.location.lower():
            return json.dumps(
                {"location": "San Francisco", "temperature": "72", "unit": weather.unit}
            )
        elif "paris" in weather.location.lower():
            return json.dumps(
                {"location": "Paris", "temperature": "22", "unit": weather.unit}
            )
        else:
            return json.dumps({"location": weather.location, "temperature": "unknown"})

    weather_tool = FunctionTool(fn=get_current_weather)
    print("tool metadata", weather_tool.definition.to_json())

    # output = get_current_weather(
    #     weather=Weather(location="San Francisco", unit="celsius")
    # )
    # print(output)
    # from langchain_core.tools import tool

    # from llama_index.core.tools import BaseTool, FunctionTool as LlamaFunctionTool

    # # weather_tool = tool(get_current_weather)
    # llama_weather_tool = LlamaFunctionTool.from_defaults(get_current_weather)
    # print("llama", llama_weather_tool.metadata.fn_schema_str)

    # from llama_index.agent.openai import OpenAIAgent
    # from llama_index.llms.openai import OpenAI

    # llm = OpenAI(model="gpt-3.5-turbo")
    # agent = OpenAIAgent.from_tools([llama_weather_tool], llm=llm, verbose=True)

    # response = agent.chat("What is the weather in San Francisco?")
    # print(response)

    template = r"""<SYS>You have these tools available:
    <TOOLS>
    {% for tool in tools %}
    {{ loop.index }}.
    {{tool}}
    ------------------------
    {% endfor %}
    </TOOLS>
    {{output_format_str}}
    </SYS>
    User: {{input_str}}
    You:
    """

    from lightrag.core.generator import Generator
    from lightrag.core.types import ModelClientType
    from lightrag.components.output_parsers import YamlOutputParser
    from lightrag.utils import enable_library_logging

    enable_library_logging()

    model_kwargs = {"model": "gpt-3.5-turbo", "temperature": 0.3, "stream": False}
    llama3_kwargs = {"model": "llama3-8b-8192"}
    function_example = Function(
        name="get_current_weather",
        kwargs={"weather": Weather(location="San Francisco", unit="celsius")},
    )
    print("function example", function_example.to_json())
    # generator, no matter what happens should generate the output
    generator = Generator(
        model_client=ModelClientType.GROQ(),
        model_kwargs=llama3_kwargs,
        template=template,
        prompt_kwargs={
            "tools": [weather_tool.definition.to_yaml()],
            "output_format_str": YamlOutputParser(
                Function  # , example=function_example
            ).format_instructions(format_type=DataClassFormatType.SIGNATURE_YAML),
            # "output_format_str": Function.to_yaml_signature(),
        },
        output_processors=YamlOutputParser(Function),
    )
    output = generator(
        prompt_kwargs={"input_str": "What is the weather in San Francisco?"}
    )
    print(output)
    generator.print_prompt()

    # json convert to its data class
    # import json

    # json_output = json.loads(json.dumps(output.data))

    # print(f"json_output: {json_output}")

    print("output.data:", output.data)
    # process the data

    # structured_output = Function(
    #     name=output.data.get("name", None),
    #     args=Weather(**output.data.get("args", None)),
    # )
    # print(structured_output)
    structured_output = Function.from_dict(output.data)
    print(f"structured_output: {structured_output}  ")
    # call the function
    function_map = {"get_current_weather": get_current_weather}
    function_name = structured_output.name
    function_args = structured_output.kwargs
    function = function_map[function_name]
    # {'name': 'get_current_weather', 'args': {'weather': {'location': 'San Francisco, CA', 'unit': 'celsius'}}}
    #  {'name': 'get_current_weather', 'args': {'location': 'San Francisco', 'unit': 'celsius'}}
    print(function(**function_args))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "get_current_weather(weather: __main__.Weather)\nGet the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "weather": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "str",
                                    "desc": "The city and state, e.g. San Francisco, CA",
                                },
                                "unit": {
                                    "type": "str",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        }
                    },
                    "required": ["weather"],
                },
            },
        }
    ]
