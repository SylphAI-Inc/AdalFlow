import inspect
import json
from typing import Callable, Dict, Any
from llama_index.llms.openai import OpenAI


def generate_schema_from_function(func: Callable) -> Dict[str, Any]:
    signature = inspect.signature(func)
    schema = {"type": "object", "properties": {}, "required": []}

    for name, parameter in signature.parameters.items():
        param_type = (
            parameter.annotation.__name__
            if parameter.annotation != inspect.Parameter.empty
            else "Any"
        )
        if parameter.default == inspect.Parameter.empty:
            schema["required"].append(name)
            schema["properties"][name] = {"type": param_type}
        else:
            schema["properties"][name] = {
                "type": param_type,
                "default": parameter.default,
            }

    return schema


def example_function(x: int, y: str = "default") -> int:
    return x


# Generate and print the schema
schema = generate_schema_from_function(example_function)
print(json.dumps(schema, indent=4))
