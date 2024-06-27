"""
Tool is LLM's extended capability which is one of the core design pattern of Agent. All tools can be wrapped in a FunctionTool class.
This helps to standardize the tool interface and metadata to communicate with the Agent.
"""

from typing import Any, Optional, Callable, Awaitable, Union
from inspect import iscoroutinefunction
import logging


from lightrag.core.types import (
    FunctionDefinition,
    FunctionOutput,
    Function,
)
from lightrag.core import Component
from lightrag.core.functional import (
    get_fun_schema,
)
from inspect import signature

AsyncCallable = Callable[..., Awaitable[Any]]

log = logging.getLogger(__name__)


def is_running_in_event_loop() -> bool:
    try:
        import asyncio

        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


FunctionType = Union[Callable[..., Any], Awaitable[Callable[..., Any]]]


# TODO: improve the support for async functions, similarly a component might be used as a tool
class FunctionTool(Component):
    __doc__ = r"""Describing and executing a function via call with arguments.


    container for a function that orchestrates the function formatting(to LLM), parsing, and execution.

    Function be used by LLM as a tool to achieve a specific task.

    Features:
    - Supports both synchronous and asynchronous functions via ``call`` and ``acall``.
    - Creates a FunctionDefinition from the function using ``get_fun_schema``.
    - Executs the function with arguments.
       [You can use Function and FunctionExpression as output format]

        - Please Parses the function call expression[FunctionExpression] into Function (name, args, kwargs).
        - call or acall, or use execute to execute the function.

         - via call with args and kwargs.
         - via eval without any context or sandboxing.
                 - via sandboxed execute directionly using ``sandbox_exec``.


    """

    def __init__(
        self,
        fn: FunctionType,
        definition: Optional[FunctionDefinition] = None,
    ):
        super().__init__()
        assert fn is not None, "fn must be provided"

        self.fn = fn
        self._is_async = iscoroutinefunction(fn)

        self.definition = definition or self._create_fn_definition()
        if self._is_async:
            log.info(f"FunctionTool: {fn} is async: {self._is_async}")

    def _create_fn_definition(self) -> FunctionDefinition:
        name = self.fn.__name__
        docstring = self.fn.__doc__
        description = f"{docstring}"
        description = f"{name}{signature(self.fn)}\n{docstring}"
        # description = f"{name}{signature(self.fn)}\n{docstring}"
        fn_parameters = get_fun_schema(name, self.fn)
        return FunctionDefinition(
            func_name=name, func_desc=description, func_parameters=fn_parameters
        )

    def call(self, *args: Any, **kwargs: Any) -> FunctionOutput:
        r"""Execute the function synchronously.

        Example:

        .. code-block:: python

            import time
            def sync_function_1():
                time.sleep(1)
                return "Function 1 completed"

            tool_1 = FunctionTool(sync_function_1)
            output = tool_1.call()
        """
        if self._is_async:
            raise ValueError("FunctionTool is asynchronous, use acall instead")
        output, error = None, None
        try:
            output = self.fn(*args, **kwargs)
        except Exception as e:
            log.error(f"Error at calling {self.fn}: {e}")
            # raise ValueError(f"Error: {e}")
            error = str(e)
        return FunctionOutput(
            name=self.definition.func_name,
            # raw_input={"args": args, "kwargs": kwargs},
            input=Function(name=self.definition.func_name, args=args, kwargs=kwargs),
            output=output,
            error=error,
        )

    async def acall(self, *args: Any, **kwargs: Any) -> FunctionOutput:
        r"""Execute the function asynchronously.

        Need to be called in an async function or using asyncio.run.

        Example:

        .. code-block:: python

            import asyncio
            async def async_function_1():
                await asyncio.sleep(1)  # Simulate async work
                return "Function 1 completed"

            async def call_async_function():
                tool_1 = FunctionTool(async_function_1)
                output = await tool_1.acall()

            asyncio.run(call_async_function())
        """
        if not self._is_async:
            raise ValueError("FunctionTool is not asynchronous, use call instead")
        output = None
        error = None
        try:
            output = await self.fn(*args, **kwargs)
        except Exception as e:
            log.error(f"Error at calling {self.fn}: {e}")
            error = str(e)

        return FunctionOutput(
            name=self.definition.func_name,
            input=Function(name=self.definition.func_name, args=args, kwargs=kwargs),
            output=output,
            error=error,
        )

    def execute(self, *args, **kwargs) -> FunctionOutput:
        r"""Execute the function synchronously or asynchronously based on the function type.

        No matter of the function type, you can run the function using both asyncio and without asyncio.
        Use it with caution as it might block the event loop.

        Example:

        .. code-block:: python

            import asyncio
            import time

            async def async_function_1():
                await asyncio.sleep(1)
                return "Function 1 completed"

            def sync_function_1():
                time.sleep(1)
                return "Function 1 completed"

            async def async_function_2():
                await asyncio.sleep(2)
                return "Function 2 completed"

            def sync_function_2():
                time.sleep(2)
                return "Function 2 completed"

            async_tool_1 = FunctionTool(async_function_1)
            sync_tool_1 = FunctionTool(sync_function_2)
            async_tool_2 = FunctionTool(async_function_2)
            sync_tool_2 = FunctionTool(sync_function_2)

            def run_sync_and_async_mix_without_wait():
                # both sync and async tool can use execute
                # sync tool can also use call
                # takes 5 seconds (1+1+2) + overhead
                start_time = time.time()
                results = [
                    async_tool_1.execute(),
                    sync_tool_1.execute(),
                    sync_tool_2.call(),
                ]
                end_time = time.time()
                print(f"run_sync_and_async_mix_without_wait time: {end_time - start_time}")
                return results

            async def run_sync_and_async_mix():
                # both sync and async tool can use execute&to_thread
                # async tool can also use acall without to_thread
                # takes a bit over 2 seconds max(2)
                start_time = time.time()
                results = await asyncio.gather(
                    asyncio.to_thread(async_tool_1.execute),
                    asyncio.to_thread(sync_tool_1.execute),
                    async_tool_2.acall(),
                )
                end_time = time.time()
                print(f"run_sync_and_async_mix time: {end_time - start_time}")
                return results

            run_sync_and_async_mix_without_wait()
            asyncio.run(run_sync_and_async_mix())
        """
        if self._is_async:
            import asyncio

            if is_running_in_event_loop():
                future = asyncio.run_coroutine_threadsafe(
                    self.acall(*args, **kwargs), asyncio.get_running_loop()
                )
                result = future.result()
            else:
                result = asyncio.run(self.acall(*args, **kwargs))
        else:
            result = self.call(*args, **kwargs)
        return result

    def __call__(self, *args, **kwargs) -> FunctionOutput:
        r"""Execute the function synchronously or asynchronously based on the function type."""
        return self.execute(*args, **kwargs)

    def _extra_repr(self) -> str:
        s = f"fn: {self.fn}, async: {self._is_async}, definition: {self.definition}"
        return s


if __name__ == "__main__":

    import asyncio
    import time

    async def async_function_1():
        await asyncio.sleep(1)
        return "Function 1 completed"

    def sync_function_1():
        time.sleep(1)
        return "Function 1 completed"

    async def async_function_2():
        await asyncio.sleep(2)
        return "Function 2 completed"

    def sync_function_2():
        time.sleep(2)
        return "Function 2 completed"

    async_tool_1 = FunctionTool(async_function_1)
    sync_tool_1 = FunctionTool(sync_function_2)
    async_tool_2 = FunctionTool(async_function_2)
    sync_tool_2 = FunctionTool(sync_function_2)

    def run_sync_and_async_mix_without_wait():
        # both sync and async tool can use execute
        # sync tool can also use call
        # takes 5 seconds (1+1+2) + overhead
        start_time = time.time()
        results = [
            async_tool_1.execute(),
            sync_tool_1.execute(),
            sync_tool_2.call(),
        ]
        end_time = time.time()
        print(f"run_sync_and_async_mix_without_wait time: {end_time - start_time}")
        return results

    async def run_sync_and_async_mix():
        # both sync and async tool can use execute&to_thread
        # async tool can also use acall without to_thread
        # takes a bit over 2 seconds max(2)
        start_time = time.time()
        results = await asyncio.gather(
            asyncio.to_thread(async_tool_1.execute),
            asyncio.to_thread(sync_tool_1.execute),
            async_tool_2.acall(),
        )
        end_time = time.time()
        print(f"run_sync_and_async_mix time: {end_time - start_time}")
        return results

    run_sync_and_async_mix_without_wait()
    asyncio.run(run_sync_and_async_mix())
    # import asyncio
    # import time

    # def sync_function_1():
    #     time.sleep(1)
    #     return "Function 1 completed"

    # def sync_function_2():
    #     time.sleep(2)
    #     return "Function 2 completed"

    # def sync_function_3():
    #     time.sleep(1.5)
    #     return "Function 3 completed"

    # # Define asynchronous functions
    # async def async_function_1():
    #     await asyncio.sleep(1)  # Simulate async work
    #     return "Function 1 completed"

    # async def async_function_2():
    #     await asyncio.sleep(2)  # Simulate async work
    #     return "Function 2 completed"

    # async def async_function_3():
    #     await asyncio.sleep(1.5)  # Simulate async work
    #     return "Function 3 completed"

    # # Create tools for async functions
    # tool_1 = FunctionTool(async_function_1)
    # tool_2 = FunctionTool(async_function_2)
    # tool_3 = FunctionTool(async_function_3)

    # sync_tool_1 = FunctionTool(sync_function_1)
    # sync_tool_2 = FunctionTool(sync_function_2)
    # sync_tool_3 = FunctionTool(sync_function_3)

    # print(f"tool_1: {tool_1._is_async}")
    # print(f"tool_2: {tool_2._is_async}")
    # print(f"tool_3: {tool_3._is_async}")

    # # Function to run and measure time using acall
    # async def run_with_acall():
    #     start_time = time.time()
    #     results = await asyncio.gather(tool_1.acall(), tool_2.acall(), tool_3.acall())
    #     end_time = time.time()
    #     print("acall Results:", [result.output for result in results])
    #     print("acall Time:", end_time - start_time)

    # def run_with_call():
    #     start_time = time.time()
    #     results = [sync_tool_1.call(), sync_tool_2.call(), sync_tool_3.call()]
    #     end_time = time.time()
    #     print("call Results:", [result.output for result in results])
    #     print("call Time:", end_time - start_time)

    # # Function to run and measure time using execute
    # async def run_with_execute():
    #     start_time = time.time()
    #     results = await asyncio.gather(
    #         asyncio.to_thread(tool_1),
    #         asyncio.to_thread(tool_2),
    #         asyncio.to_thread(tool_3),
    #     )
    #     end_time = time.time()
    #     print("execute Results:", [result.output for result in results])
    #     print("execute Time:", end_time - start_time)

    # async def run_sync_as_async():  # using execute
    #     start_time = time.time()
    #     results = await asyncio.gather(
    #         asyncio.to_thread(sync_tool_1),
    #         asyncio.to_thread(sync_tool_2),
    #         asyncio.to_thread(sync_tool_3),
    #     )
    #     end_time = time.time()
    #     print("execute sync as async Results:", [result.output for result in results])
    #     print("execute sync as async Time:", end_time - start_time)

    # def run_sync_and_async_mix_without_asyncio():  # using execute
    #     start_time = time.time()
    #     results = [sync_tool_1.execute(), sync_tool_2.call(), tool_3.execute()]
    #     end_time = time.time()
    #     print(
    #         "execute sync and async mix without asyncio Results:",
    #         [result.output for result in results],
    #     )
    #     print("execute sync and async mix without asyncio Time:", end_time - start_time)

    # async def run_sync_and_async_mix():  # using execute
    #     start_time = time.time()
    #     results = await asyncio.gather(
    #         asyncio.to_thread(sync_tool_1.execute),
    #         asyncio.to_thread(tool_2.execute),
    #         tool_3.acall(),
    #     )

    #     end_time = time.time()
    #     print(
    #         "execute sync and async mix Results:", [result.output for result in results]
    #     )
    #     print("execute sync and async mix Time:", end_time - start_time)

    # # you can run the async function without asyncio and the effect is the same as sync function
    # def run_with_execute_sync():
    #     start_time = time.time()
    #     results = [tool_1(), tool_2(), tool_3()]
    #     end_time = time.time()
    #     print("execute without asyncio Results:", [result.output for result in results])
    #     print("execute without asyncio Time:", end_time - start_time)

    # # Run tests
    # async def main():
    #     # print("Testing with call:")
    #     # run_with_call()
    #     # print("Testing with acall:")
    #     # await run_with_acall()

    #     # print("\nTesting with execute:")
    #     # await run_with_execute()

    #     # print("\nTesting with execute sync as async:")
    #     # await run_sync_as_async()

    #     print("\nTesting with execute sync and async mix:")
    #     await run_sync_and_async_mix()

    # # call and measure
    # asyncio.run(main())

    # # print("\nTesting with execute without asyncio:")
    # # run_with_execute_sync()
    # print("\nTesting with execute sync and async mix without asyncio:")
    # run_sync_and_async_mix_without_asyncio()


# from lightrag.utils import setup_env  # noqa
# from lightrag.utils import enable_library_logging

# enable_library_logging()

# def get_current_weather_1(location: str, unit: str = "fahrenheit"):
#     """Get the current weather in a given location"""
#     if "tokyo" in location.lower():
#         return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
#     elif "san francisco" in location.lower():
#         return json.dumps(
#             {"location": "San Francisco", "temperature": "72", "unit": unit}
#         )
#     elif "paris" in location.lower():
#         return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
#     else:
#         return json.dumps({"location": location, "temperature": "unknown"})

# from lightrag.core.base_data_class import DataClassFormatType
# from dataclasses import dataclass, field

# @dataclass
# class Weather:
#     location: str = field(
#         metadata={"desc": "The city and state, e.g. San Francisco, CA"}
#     )
#     unit: str = field(metadata={"enum": ["celsius", "fahrenheit"]})

# def get_current_weather(weather: Union[Weather, Dict[str, Dict]]):
#     """Get the current weather in a given location"""
#     # LLM will work better with str and dict inputs.
#     if isinstance(weather, str):
#         weather = json.loads(weather)
#     if isinstance(weather, dict):
#         weather = Weather(**weather)

#     if "tokyo" in weather.location.lower():
#         return json.dumps(
#             {"location": "Tokyo", "temperature": "10", "unit": weather.unit}
#         )
#     elif "san francisco" in weather.location.lower():
#         return json.dumps(
#             {"location": "San Francisco", "temperature": "72", "unit": weather.unit}
#         )
#     elif "paris" in weather.location.lower():
#         return json.dumps(
#             {"location": "Paris", "temperature": "22", "unit": weather.unit}
#         )
#     else:
#         return json.dumps({"location": weather.location, "temperature": "unknown"})

# weather_tool = FunctionTool(fn=get_current_weather)
# print("tool metadata", weather_tool.definition.to_json())

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


# from lightrag.core.generator import Generator
# from lightrag.core.types import ModelClientType
# from lightrag.components.output_parsers import YamlOutputParser
# from lightrag.utils import enable_library_logging

# enable_library_logging()

# model_kwargs = {"model": "gpt-3.5-turbo", "temperature": 0.3, "stream": False}
# llama3_kwargs = {"model": "llama3-8b-8192"}
# function_example = Function(
#     name="get_current_weather",
#     kwargs={"weather": Weather(location="San Francisco", unit="celsius")},
# )
# print("function example", function_example.to_json())
# # generator, no matter what happens should generate the output
# generator = Generator(
#     model_client=ModelClientType.GROQ(),
#     model_kwargs=llama3_kwargs,
#     template=template,
#     prompt_kwargs={
#         "tools": [weather_tool.definition.to_yaml()],
#         "output_format_str": YamlOutputParser(
#             data_class=Function  # , example=function_example
#         ).format_instructions(format_type=DataClassFormatType.SIGNATURE_YAML),
#         # "output_format_str": Function.to_yaml_signature(),
#     },
#     output_processors=YamlOutputParser(Function),
# )
# output = generator(
#     prompt_kwargs={"input_str": "What is the weather in San Francisco?"}
# )
# print(output)
# generator.print_prompt()

# # json convert to its data class
# # import json

# # json_output = json.loads(json.dumps(output.data))

# # print(f"json_output: {json_output}")

# print("output.data:", output.data)
# # process the data

# # structured_output = Function(
# #     name=output.data.get("name", None),
# #     args=Weather(**output.data.get("args", None)),
# # )
# # print(structured_output)
# structured_output = Function.from_dict(output.data)
# print(f"structured_output: {structured_output}  ")
# # call the function
# function_map = {"get_current_weather": get_current_weather}
# function_name = structured_output.name
# function_args = structured_output.kwargs
# function = function_map[function_name]
# # {'name': 'get_current_weather', 'args': {'weather': {'location': 'San Francisco, CA', 'unit': 'celsius'}}}
# #  {'name': 'get_current_weather', 'args': {'location': 'San Francisco', 'unit': 'celsius'}}
# print(function(**function_args))
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_current_weather",
#             "description": "get_current_weather(weather: __main__.Weather)\nGet the current weather in a given location",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "weather": {
#                         "type": "object",
#                         "properties": {
#                             "location": {
#                                 "type": "str",
#                                 "desc": "The city and state, e.g. San Francisco, CA",
#                             },
#                             "unit": {
#                                 "type": "str",
#                                 "enum": ["celsius", "fahrenheit"],
#                             },
#                         },
#                         "required": ["location", "unit"],
#                     }
#                 },
#                 "required": ["weather"],
#             },
#         },
#     }
# ]
