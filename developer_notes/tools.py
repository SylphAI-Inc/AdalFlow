from dataclasses import dataclass, field
from typing import List
import numpy as np
import time
import asyncio
from lightrag.utils import setup_env  # noqa
from lightrag.core import Component, DataClass
from lightrag.core.types import Function, FunctionExpression
from lightrag.core.tool_manager import ToolManager
from lightrag.components.output_parsers import JsonOutputParser

from lightrag.core.generator import Generator
from lightrag.core.types import ModelClientType


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    time.sleep(2)
    return a * b


def add(a: int, b: int) -> int:
    """Add two numbers."""
    time.sleep(3)
    return a + b


async def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    await asyncio.sleep(3)
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


@dataclass  # TODO: data class schema, need to go all the way down to the subclass
class MultipleFunctionDefinition(DataClass):
    a: Point = field(metadata={"desc": "First number"})
    b: int = field(metadata={"desc": "Second number"})


# optionally to define schma yourself, this can be used to generate FunctionDefinition
print(MultipleFunctionDefinition.to_schema_str())
# use function tool

template = r"""<SYS>You have these tools available:
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
</SYS>
User: {{input_str}}
You:
"""

template_with_context = r"""<SYS>You have these tools available:
{% if tools %}
<TOOLS>
{% for tool in tools %}
{{ loop.index }}.
{{tool}}
------------------------
{% endfor %}
</TOOLS>
{% endif %}
<CONTEXT>
Your function expression also have access to these context:
{{context_str}}
</CONTEXT>
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
</SYS>
User: {{input_str}}
You:
"""

multple_function_call_template = r"""<SYS>You have these tools available:
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
Here is how you call one function.
{{output_format_str}}
-Always return a List using `[]` of the above JSON objects, even if its just one item.
</OUTPUT_FORMAT>
<SYS>
{{input_str}}
You:
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
functions = [multiply, add, divide, search, numpy_sum, add_points]


class FunctionCall(Component):
    def __init__(self):
        super().__init__()

    def prepare_single_function_call_generator(self):
        tool_manager = ToolManager(tools=functions)
        func_parser = JsonOutputParser(
            data_class=Function, exclude_fields=["thought", "args"]
        )

        model_kwargs = {"model": "gpt-3.5-turbo"}
        prompt_kwargs = {
            "tools": tool_manager.yaml_definitions,
            "output_format_str": func_parser.format_instructions(),
        }
        generator = Generator(
            model_client=ModelClientType.OPENAI(),
            model_kwargs=model_kwargs,
            template=template,
            prompt_kwargs=prompt_kwargs,
            output_processors=func_parser,
        )
        generator.print_prompt(**prompt_kwargs)
        return generator, tool_manager

    def run_function_call(self, generator: Generator, tool_manager: ToolManager):

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


class FunctionCallWithFunctionExpression(Component):
    def __init__(self):
        super().__init__()

    def prepare_single_function_call_generator(self):
        tool_manager = ToolManager(
            tools=functions,
            additional_context={
                "x": x,
                "y": 0,
                "np.array": np.array,
                "np": np,
                "Point": Point,
            },
        )
        func_parser = JsonOutputParser(
            data_class=FunctionExpression, exclude_fields=["thought", "args"]
        )
        instructions = func_parser.format_instructions()
        print(instructions)

        model_kwargs = {"model": "gpt-4o"}
        prompt_kwargs = {
            "tools": tool_manager.yaml_definitions,
            "output_format_str": func_parser.format_instructions(),
            "context_str": tool_manager._additional_context,
        }
        generator = Generator(
            model_client=ModelClientType.OPENAI(),
            model_kwargs=model_kwargs,
            template=template_with_context,
            prompt_kwargs=prompt_kwargs,
            output_processors=func_parser,
        )
        generator.print_prompt(**prompt_kwargs)
        return generator, tool_manager

    def run_function_call(self, generator: Generator, tool_manager: ToolManager):
        start_time = time.time()
        for idx, query in enumerate(queries):
            prompt_kwargs = {"input_str": query}
            print(f"\n{idx} Query: {query}")
            print(f"{'-'*50}")
            try:
                result = generator(prompt_kwargs=prompt_kwargs)
                # print(f"LLM raw output: {result.raw_response}")
                func_expr = FunctionExpression.from_dict(result.data)
                print(f"Function_expr: {func_expr}")
                # func: Function = tool_manager.parse_func_expr(func_expr)
                # func_output = tool_manager.execute_func(func)
                # or
                # func_output = tool_manager.execute_func_expr_via_sandbox(func_expr)
                # or
                func_output = tool_manager.execute_func_expr_via_eval(func_expr)
                print(f"Function output: {func_output}")
            except Exception as e:
                print(
                    f"Failed to execute the function for query: {query}, func: {result.data}, error: {e}"
                )
        end_time = time.time()
        print(f"Total time taken: {end_time - start_time :.2f} seconds")

    async def run_async_function_call(self, generator, tool_manager):
        answers = []
        start_time = time.time()
        tasks = []
        for idx, query in enumerate(queries):
            tasks.append(self.process_query(idx, query, generator, tool_manager))

        results = await asyncio.gather(*tasks)
        answers.extend(results)
        end_time = time.time()
        print(f"Total time taken: {end_time - start_time :.2f} seconds")
        return answers

    async def process_query(self, idx, query, generator, tool_manager: ToolManager):
        print(f"\n{idx} Query: {query}")
        print(f"{'-'*50}")
        try:
            result = generator(prompt_kwargs={"input_str": query})
            func_expr = FunctionExpression.from_dict(result.data)
            print(f"Function_expr: {func_expr}")
            # func = tool_manager.parse_func_expr(func_expr)
            # func_output = await tool_manager.execute_func(func)
            # or
            func_output = await tool_manager.execute_func_expr(func_expr)

            print(f"Function output: {func_output}")
            return func_output
        except Exception as e:
            print(
                f"Failed to execute the function for query: {query}, func: {result.data}, error: {e}"
            )
            return None


class MultiFunctionCallWithFunctionExpression(Component):
    def __init__(self):
        super().__init__()

    def prepare_single_function_call_generator(self):
        tool_manager = ToolManager(
            tools=functions,
            additional_context={
                "x": x,
                "y": 0,
                "np.array": np.array,
                "np": np,
                "Point": Point,
            },
        )
        example = FunctionExpression.from_function(
            func=add_points, p1=Point(x=1, y=2), p2=Point(x=3, y=4)
        )
        func_parser = JsonOutputParser(
            data_class=FunctionExpression,
            examples=[example],
            exclude_fields=["thought"],
        )
        instructions = func_parser.format_instructions()
        print(instructions)

        model_kwargs = {"model": "gpt-4o"}
        prompt_kwargs = {
            "tools": tool_manager.yaml_definitions,
            "output_format_str": func_parser.format_instructions(),
        }
        generator = Generator(
            model_client=ModelClientType.OPENAI(),
            model_kwargs=model_kwargs,
            template=multple_function_call_template,
            prompt_kwargs=prompt_kwargs,
            output_processors=func_parser,
        )
        generator.print_prompt(**prompt_kwargs)
        return generator, tool_manager

    def run_function_call(self, generator: Generator, tool_manager: ToolManager):
        start_time = time.time()
        for idx in range(0, len(queries), 2):
            query = " and ".join(queries[idx : idx + 2])
            prompt_kwargs = {"input_str": query}
            print(f"\n{idx} Query: {query}")
            print(f"{'-'*50}")
            try:
                result = generator(prompt_kwargs=prompt_kwargs)
                print(f"LLM raw output: {result.raw_response}")
                func_expr: List[FunctionExpression] = [
                    FunctionExpression.from_dict(item) for item in result.data
                ]
                print(f"Function_expr: {func_expr}")
                for expr in func_expr:
                    func_output = tool_manager.execute_func_expr_via_eval(expr)
                    print(f"Function output: {func_output}")
            except Exception as e:
                print(
                    f"Failed to execute the function for query: {query}, func: {result.data}, error: {e}"
                )
        end_time = time.time()
        print(f"Total time taken: {end_time - start_time :.2f} seconds")


if __name__ == "__main__":
    # fc = FunctionCall()
    # generator, tool_manager = fc.prepare_single_function_call_generator()
    # fc.run_function_call(generator, tool_manager)

    # fc = FunctionCallWithFunctionExpression()
    # generator, tool_manager = fc.prepare_single_function_call_generator()
    # fc.run_function_call(generator, tool_manager)  # 15.92s
    # asyncio.run(fc.run_async_function_call(generator, tool_manager))  # 7.8s

    output = eval("add(a=y, b=5)", {"y": 3, "add": add})
    print(output)

    mul_fc = MultiFunctionCallWithFunctionExpression()
    generator, tool_manager = mul_fc.prepare_single_function_call_generator()
    mul_fc.run_function_call(generator, tool_manager)
