"""
Tool is LLM's extended capability which is one of the core design pattern of Agent. All tools can be wrapped in a FunctionTool class.
This helps to standardize the tool interface and metadata to communicate with the Agent.
"""

from typing import Any, Optional, Dict, Callable, Awaitable, get_type_hints, Union
from inspect import iscoroutinefunction, signature, Parameter

from dataclasses import dataclass, fields, is_dataclass, field

import json

from lightrag.core.types import ToolMetadata, ToolOutput
from lightrag.core import Component

AsyncCallable = Callable[..., Awaitable[Any]]


def get_fun_schema_1(name: str, func: Callable[..., Any]) -> Dict[str, Any]:
    r"""Get the schema of a function.
    Examples:
    def example_function(x: int, y: str = "default") -> int:
        return x
    schema = get_fun_schema("example_function", example_function)
    print(json.dumps(schema, indent=4))
    # Output:
    {
        "type": "object",
        "properties": {
            "x": {
                "type": "int"
            },
            "y": {
                "type": "str",
                "default": "default"
            }
        },
        "required": [
            "x"
        ]
    }
    """
    sig = signature(func)
    schema = {"type": "object", "properties": {}, "required": []}
    type_hints = get_type_hints(func)

    for name, parameter in sig.parameters.items():
        print(f"name: {name}, parameter: {parameter}    {parameter.annotation}")
        param_type = (
            parameter.annotation.__name__
            if parameter.annotation != Parameter.empty
            else "Any"
        )
        # add type and default value if it exists
        if parameter.default == Parameter.empty:
            schema["required"].append(name)
            schema["properties"][name] = {"type": param_type}
        else:
            schema["properties"][name] = {
                "type": param_type,
                "default": parameter.default,
            }
        # allow nested dataclasses
        if is_dataclass(type_hints[name]):
            print(f"type_hints[name]: {type_hints[name], name}")
            for field_ in fields(type_hints[name]):
                # format anything in the metadata
                if "default" in field_.metadata:
                    schema["properties"][name]["default"] = field_.metadata["default"]
                for meta_field in field_.metadata:
                    if meta_field != "default":
                        schema["properties"][name][meta_field] = field_.metadata[
                            meta_field
                        ]

        # add definitions if nested model exists
        if hasattr(parameter.annotation, "__annotations__"):
            schema["definitions"] = {name: get_fun_schema(name, parameter.annotation)}

    return schema


def get_fun_schema(name: str, func: Callable[..., Any]) -> Dict[str, Any]:
    r"""Get the schema of a function.
    Examples:
    def example_function(x: int, y: str = "default") -> int:
        return x
    schema = get_fun_schema("example_function", example_function)
    print(json.dumps(schema, indent=4))
    # Output:
    {
        "type": "object",
        "properties": {
            "x": {
                "type": "int"
            },
            "y": {
                "type": "str",
                "default": "default"
            }
        },
        "required": [
            "x"
        ]
    }
    """
    sig = signature(func)
    schema = {"type": "object", "properties": {}, "required": []}
    type_hints = get_type_hints(func)

    for param_name, parameter in sig.parameters.items():
        param_type = (
            parameter.annotation.__name__
            if parameter.annotation != Parameter.empty
            else "Any"
        )
        if parameter.default == Parameter.empty:
            schema["required"].append(param_name)
            schema["properties"][param_name] = {"type": param_type}
        else:
            schema["properties"][param_name] = {
                "type": param_type,
                "default": parameter.default,
            }

        # Check if the parameter is a dataclass
        if is_dataclass(type_hints[param_name]):
            schema["properties"][param_name] = get_dataclass_schema(
                type_hints[param_name]
            )

    return schema


def get_dataclass_schema(cls):
    """Generate schema for a dataclass."""
    schema = {"type": "object", "properties": {}, "required": []}
    for field_ in fields(cls):
        field_schema = {"type": field_.type.__name__}
        if field_.default != field_.default_factory:
            field_schema["default"] = field_.default
        if field_.metadata:
            field_schema.update(field_.metadata)
        schema["properties"][field_.name] = field_schema
        if field_.default == field_.default_factory:
            schema["required"].append(field_.name)

    return schema


# def get_fun_schema(name: str, func: Callable[..., Any]) -> Dict[str, Any]:
#     r"""Get the schema of a function.
#     Examples:
#     def example_function(x: int, y: str = "default") -> int:
#         return x
#     schema = get_fun_schema("example_function", example_function)
#     print(json.dumps(schema, indent=4))
#     # Output:
#     {
#         "type": "object",
#         "properties": {
#             "x": {
#                 "type": "int"
#             },
#             "y": {
#                 "type": "str",
#                 "default": "default"
#             }
#         },
#         "required": [
#             "x"
#         ]
#     }
#     """
#     sig = signature(func)
#     schema = {"type": "object", "properties": {}, "required": []}
#     type_hints = get_type_hints(func)

#     for name, parameter in sig.parameters.items():
#         print(f"name: {name}, parameter: {parameter}    {parameter.annotation}")
#         param_type = (
#             parameter.annotation.__name__
#             if parameter.annotation != Parameter.empty
#             else "Any"
#         )
#         if parameter.default == Parameter.empty:
#             schema["required"].append(name)
#             schema["properties"][name] = {"type": param_type}
#         else:
#             schema["properties"][name] = {
#                 "type": param_type,
#                 "default": parameter.default,
#             }
#         # allow nested dataclasses
#         if is_dataclass(type_hints[name]):
#             print(f"type_hints[name]: {type_hints[name]}")
#             for field_ in fields(type_hints[name]):
#                 # format anything in the metadata
#                 if "default" in field_.metadata:
#                     schema["properties"][name]["default"] = field_.metadata["default"]
#                 for meta_field in field_.metadata:
#                     if meta_field != "default":
#                         schema["properties"][name][meta_field] = field_.metadata[
#                             meta_field
#                         ]

#         # add definitions if nested model exists
#         if hasattr(parameter.annotation, "__annotations__"):
#             schema["definitions"] = {name: get_fun_schema(name, parameter.annotation)}

#     return schema


# @dataclass
class FunctionTool(Component):
    __doc__ = r"""
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
        metadata: Optional[ToolMetadata] = None,
    ):
        super().__init__()
        assert fn is not None, "fn must be provided"

        self.fn = fn
        self._is_async = iscoroutinefunction(fn)
        print(f"self._is_async: {self._is_async}")

        self.metadata = metadata or self._create_metadata()

    def _create_metadata(self) -> ToolMetadata:
        name = self.fn.__name__
        docstring = self.fn.__doc__
        description = f"{name}{signature(self.fn)}\n{docstring}"
        fn_parameters = get_fun_schema(name, self.fn)
        return ToolMetadata(
            name=name, description=description, parameters=fn_parameters
        )

    # @classmethod
    # def from_defaults(
    #     cls,
    #     fn: Optional[
    #         Callable[..., Any]
    #     ] = None,  # at least one of fn or async_fn must be provided
    #     async_fn: Optional[AsyncCallable] = None,
    #     name: Optional[str] = None,
    #     description: Optional[
    #         str
    #     ] = None,  # if not provided, use function name, signature and docstring
    #     tool_metadata: Optional[ToolMetadata] = None,
    # ) -> "FunctionTool":
    #     if tool_metadata is None:
    #         name = name or fn.__name__
    #         docstring = fn.__doc__
    #         # sample_function(x, y, user: tests.test_tool.User = User(id=1, name='John'))
    #         # two numbers together and returns the sum.
    #         description = description or f"{name}{signature(fn)}\n{docstring}"

    #         # fn_parameters are more readable than the above name, signature and docstring combination
    #         fn_parameters = get_fun_schema(name, fn)

    #         tool_metadata = ToolMetadata(
    #             name=name, description=description, parameters=fn_parameters
    #         )
    #     return cls(fn=fn, metadata=tool_metadata, async_fn=async_fn)

    # def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
    #     return self.call(*args, **kwargs)

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

    from lightrag.core.base_data_class import DataClass
    from dataclasses import dataclass, field

    @dataclass
    class Weather:
        location: str = field(
            metadata={"desc": "The city and state, e.g. San Francisco, CA"}
        )
        unit: str = field(metadata={"enum": ["celsius", "fahrenheit"]})

    def get_current_weather(weather: Weather):
        """Get the current weather in a given location"""
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

    weather_tool = FunctionTool(fn=get_current_weather_1)
    print(weather_tool.metadata.to_json())

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
    @dataclass
    class Function(DataClass):
        name: str = field(metadata={"desc": "The name of the function"})
        args: Dict[str, Any] = field(metadata={"desc": "The arguments of the function"})

    template = r"""<SYS>You have these tools available:
    <TOOLS>
    {% for tool in tools %}
    {{ loop.index }}. ToolName: {{ tool.metadata.name }}
        Tool Description: {{ tool.metadata.description }}
        Tool Parameters: {{ tool.metadata.fn_schema_str }} {#tool args can be misleading, especially if we already have type hints and docstring in the function#}
    __________
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

    generator = Generator(
        model_client=ModelClientType.OPENAI(),
        model_kwargs=model_kwargs,
        template=template,
        prompt_kwargs={
            "tools": [weather_tool],
            "output_format_str": YamlOutputParser(Function).format_instructions(),
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
    function_map = {"get_current_weather_1": get_current_weather_1}
    function_name = structured_output.name
    function_args = structured_output.args
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
