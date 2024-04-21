"""
test file: tests/test_tool.py

Different from llamaindex which defines these types of tools:
- FunctionTool
- RetrieverTool
- QueryEngineTool
-...
Llamaindex: BaseTool->AsyncBaseTool->FunctionTool
Our tool is an essential callable object (similar to the function tool) that you can wrap in any other parts such as retriever, generator in.
TO support:
- sync tool 
- async tool
TODO: to observe and improve the mix of sync and async tools in the future.
How can we know after the llm call that a function tool is sync or async?
"""

from typing import Any, Optional, Type, Dict, Callable, Awaitable, Tuple
from inspect import iscoroutinefunction, signature, Parameter
from pydantic import BaseModel, create_model
import json

AsyncCallable = Callable[..., Awaitable[Any]]


##############################################
# Tool data classes, using BaseModel to auto-generate schema
# Simplified version of LlamaIndex's BaseTool
##############################################
class ToolOutput(BaseModel):
    str_content: Optional[str] = None  # Initially allow str_content to be optional

    name: Optional[str] = None
    raw_input: Dict[str, Any]
    raw_output: Any

    def __init__(self, **data):
        if "str_content" not in data or data["str_content"] is None:
            data["str_content"] = str(data["raw_output"])
        super().__init__(**data)

    def __str__(self) -> str:
        return str(self.str_content)


class ToolMetadata(BaseModel):
    """
    Metadata for a tool. Can be passed to LLM for tool registration.
    """

    description: str
    name: Optional[str] = None
    parameters: Dict[str, Any] = {}

    def get_parameters_dict(self) -> dict:
        parameters = {
            k: v
            for k, v in self.parameters.items()
            if k in ["type", "properties", "required", "definitions"]
        }
        return parameters

    @property
    def tool_str(self) -> str:
        """
        Return a string representation of the tool.
        """
        return self.description

    @property
    def fn_schema_str(self) -> str:
        parameters = self.get_parameters_dict()
        return json.dumps(parameters)

    def get_name(self) -> str:
        if self.name is None:
            raise ValueError("name is None.")
        return self.name


##############################################
# Helper functions
##############################################
def get_fun_schema(name: str, func: Callable[..., Any]) -> Dict[str, Any]:
    """
    Get a schema for a function.
    Example:
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

    for name, parameter in sig.parameters.items():
        param_type = (
            parameter.annotation.__name__
            if parameter.annotation != Parameter.empty
            else "Any"
        )
        if parameter.default == Parameter.empty:
            schema["required"].append(name)
            schema["properties"][name] = {"type": param_type}
        else:
            schema["properties"][name] = {
                "type": param_type,
                "default": parameter.default,
            }
        # add definitions if nested model exists
        if hasattr(parameter.annotation, "__annotations__"):
            schema["definitions"] = {name: get_fun_schema(name, parameter.annotation)}

    return schema


##############################################
# FunctionTool
##############################################
class FunctionTool:
    """
    There is almost no need to customize a FunctionTool, but you can do so if you want to.
    Support both positional and keyword arguments.
    NOTE:
    - at least one of fn or async_fn must be provided.
    - When both are provided, sync (call) will be used in __call__.
    """

    def __init__(
        self,
        metadata: ToolMetadata,
        fn: Optional[
            Callable[..., Any]
        ] = None,  # at least one of fn or async_fn must be provided
        async_fn: Optional[AsyncCallable] = None,
    ) -> None:
        self._fn = None
        self._async_fn = None
        if fn:
            self._fn = fn
        elif async_fn:
            if not iscoroutinefunction(async_fn):
                raise ValueError("async_fn must be an asynchronous function")
            self._async_fn = async_fn
        # else:
        #     raise ValueError("At least one of fn or async_fn must be provided")

        self._metadata = metadata

    @classmethod
    def from_defaults(
        cls,
        fn: Optional[
            Callable[..., Any]
        ] = None,  # at least one of fn or async_fn must be provided
        async_fn: Optional[AsyncCallable] = None,
        name: Optional[str] = None,
        description: Optional[
            str
        ] = None,  # if not provided, use function name, signature and docstring
        tool_metadata: Optional[ToolMetadata] = None,
    ) -> "FunctionTool":
        if tool_metadata is None:
            name = name or fn.__name__
            docstring = fn.__doc__
            # sample_function(x, y, user: tests.test_tool.User = User(id=1, name='John'))
            # two numbers together and returns the sum.
            description = description or f"{name}{signature(fn)}\n{docstring}"

            # fn_parameters are more readable than the above name, signature and docstring combination
            fn_parameters = get_fun_schema(name, fn)

            tool_metadata = ToolMetadata(
                name=name, description=description, parameters=fn_parameters
            )
        return cls(fn=fn, metadata=tool_metadata, async_fn=async_fn)

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    @property
    def fn(self) -> Callable[..., Any]:
        return self._fn

    @property
    def async_fn(self) -> AsyncCallable:
        return self._async_fn

    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        return self.call(*args, **kwargs)

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        tool_output = self._fn(*args, **kwargs)
        return ToolOutput(
            tool_name=self.metadata.name,
            raw_input={"args": args, "kwargs": kwargs},
            raw_output=tool_output,
        )

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        tool_output = await self._async_fn(*args, **kwargs)
        return ToolOutput(
            tool_name=self.metadata.name,
            raw_input={"args": args, "kwargs": kwargs},
            raw_output=tool_output,
        )
