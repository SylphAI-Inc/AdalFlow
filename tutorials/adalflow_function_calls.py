"""
This script demonstrates the usage of AdalFlow's Tool And ToolManager functionality.
Tutorial link: https://adalflow.sylph.ai/tutorials/tool_helper.html
"""

from dataclasses import dataclass
import numpy as np
from typing import List
import time
import asyncio
from adalflow.utils import printc

from adalflow.core.tool_manager import ToolManager
from adalflow.core.types import FunctionExpression, Function


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    time.sleep(1)
    return a * b


def add(a: int, b: int) -> int:
    """Add two numbers."""
    time.sleep(1)
    return a + b


async def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    await asyncio.sleep(1)
    return float(a) / b


async def search(query: str) -> List[str]:
    """Search for query and return a list of results."""
    await asyncio.sleep(1)
    return ["result1" + query, "result2" + query]


def numpy_sum(arr: np.ndarray) -> float:
    """Sum the elements of an array."""
    return np.sum(arr)


x = 2


@dataclass
class Point:
    x: int
    y: int


def add_points(p1: Point, p2: Point) -> Point:
    return Point(p1.x + p2.x, p1.y + p2.y)


def learn_function_tool():
    from adalflow.core.func_tool import FunctionTool

    functions = [multiply, add, divide, search, numpy_sum, add_points]
    tools = [FunctionTool(fn=fn) for fn in functions]
    for tool in tools:
        printc(f"Function: {tool}")

    definition_dict = tools[-2].definition.to_dict()
    printc(f"Definition Dict: {definition_dict}", color="yellow")


template = r"""<START_OF_SYS_PROMPT>You have these tools available:
{% if tools %}
<TOOLS>
{% for tool in tools %}
{{ loop.index }}.
{{tool}}
------------------------
{% endfor %}
</TOOLS>
{% endif %}
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
<END_OF_SYS_PROMPT>
<START_OF_USER>: {{input_str}}<END_OF_USER>
"""

queries = [
    "add 2 and 3",
    "search for something",
    "add points (1, 2) and (3, 4)",
    "sum numpy array with arr = np.array([[1, 2], [3, 4]])",
    "multiply 2 with local variable x",
    "divide 2 by 3",
    "Add 5 to variable y",
]


def learn_tool_manager():

    from adalflow.core.tool_manager import ToolManager

    tools = [multiply, add, divide, search, numpy_sum, add_points]

    tool_manager = ToolManager(tools=tools)
    printc(f"Tool Manager: {tool_manager}")

    from adalflow.core.prompt_builder import Prompt

    prompt = Prompt(template=template)
    small_tool_manager = ToolManager(tools=tools[:2])

    renered_prompt = prompt(tools=small_tool_manager.yaml_definitions)
    printc(f"Prompt: {renered_prompt}", color="yellow")

    # get output format with function

    output_data_class = Function
    output_format_str = output_data_class.to_json_signature(exclude=["thought", "args"])

    renered_prompt = prompt(output_format_str=output_format_str)
    printc(renered_prompt)

    # get output format with functionexperession
    from adalflow.core.types import FunctionExpression

    output_data_class = FunctionExpression
    output_format_str = output_data_class.to_json_signature(exclude=["thought"])
    printc(prompt(output_format_str=output_format_str), color="yellow")

    # output format instruction

    from adalflow.components.output_parsers import JsonOutputParser

    func_parser = JsonOutputParser(
        data_class=Function, exclude_fields=["thought", "args"]
    )
    instructions = func_parser.format_instructions()
    printc(f"Format instructions: {instructions}")


def learn_run_generator_with_function_end_to_end():
    from adalflow.core.tool_manager import ToolManager
    from adalflow.core.generator import Generator
    from adalflow.components.model_client import OpenAIClient

    from adalflow.components.output_parsers import JsonOutputParser

    func_parser = JsonOutputParser(
        data_class=Function, exclude_fields=["thought", "args"]
    )

    tools = [multiply, add, divide, search, numpy_sum, add_points]

    tool_manager = ToolManager(tools=tools)

    model_kwargs = {"model": "gpt-3.5-turbo"}
    prompt_kwargs = {
        "tools": tool_manager.yaml_definitions,
        "output_format_str": func_parser.format_instructions(),
    }
    generator = Generator(
        model_client=OpenAIClient(),
        model_kwargs=model_kwargs,
        template=template,
        prompt_kwargs=prompt_kwargs,
        output_processors=func_parser,
    )
    # two will fail which is fine.
    for idx, query in enumerate(queries):
        prompt_kwargs = {"input_str": query}
        print(f"\n{idx} Query: {query}")
        print(f"{'-'*50}")
        try:
            result = generator(prompt_kwargs=prompt_kwargs)
            # print(f"LLM raw output: {result.raw_response}")
            func = Function.from_dict(result.data)
            print(f"Function: {func}")
            func_output = tool_manager.execute_func(func)
            print(f"Function output: {func_output}")
        except Exception as e:
            print(
                f"Failed to execute the function for query: {query}, func: {result.data}, error: {e}"
            )


context = r"""<CONTEXT>
Your function expression also have access to these context:
{{context_str}}
</CONTEXT>
"""


async def process_query(idx, query, generator, tool_manager: ToolManager):
    print(f"\n{idx} Query: {query}")
    print(f"{'-'*50}")
    try:
        result = generator(prompt_kwargs={"input_str": query})
        func_expr = FunctionExpression.from_dict(result.data)
        print(f"Function_expr: {func_expr}")
        func = tool_manager.parse_func_expr(func_expr)
        func_output = await tool_manager.execute_func_async(func)
        print(f"Function output: {func_output}")
        return func_output
    except Exception as e:
        print(
            f"Failed to execute the function for query: {query}, func: {result.data}, error: {e}"
        )
        return None


async def run_async_function_call(generator, tool_manager):
    answers = []
    start_time = time.time()
    tasks = []
    for idx, query in enumerate(queries):
        tasks.append(process_query(idx, query, generator, tool_manager))

    results = await asyncio.gather(*tasks)
    answers.extend(results)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time :.2f} seconds")
    return answers


def learn_run_generator_with_function_expressions_end_to_end():

    from adalflow.core.generator import Generator
    from adalflow.components.model_client import OpenAIClient
    from adalflow.components.output_parsers import JsonOutputParser

    func_parser = JsonOutputParser(data_class=FunctionExpression)

    tools = [multiply, add, divide, search, numpy_sum, add_points]
    tool_manager = ToolManager(
        tools=tools,
        additional_context={"x": x, "y": 0, "np.array": np.array, "np": np},
    )

    model_kwargs = {"model": "gpt-3.5-turbo"}
    prompt_kwargs = {
        "tools": tool_manager.yaml_definitions,
        "output_format_str": func_parser.format_instructions(),
    }
    generator = Generator(
        model_client=OpenAIClient(),
        model_kwargs=model_kwargs,
        template=template,
        prompt_kwargs=prompt_kwargs,
        output_processors=func_parser,
    )

    import asyncio

    asyncio.run(run_async_function_call(generator, tool_manager))


def main():
    learn_function_tool()
    learn_tool_manager()
    learn_run_generator_with_function_end_to_end()
    learn_run_generator_with_function_expressions_end_to_end()


if __name__ == "__main__":
    from adalflow.utils import setup_env

    setup_env()
    main()
