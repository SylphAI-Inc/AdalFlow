"""
Tool is LLM's extended capability which is one of the core design pattern of Agent. All tools can be wrapped in a FunctionTool class.
This helps to standardize the tool interface and metadata to communicate with the Agent.
"""

from typing import Any, Optional, Callable, Awaitable, Union
from inspect import iscoroutinefunction
import logging
import asyncio
import nest_asyncio


from adalflow.core.types import (
    FunctionDefinition,
    FunctionOutput,
    Function,
)
from adalflow.core import Component
from adalflow.core.functional import (
    get_fun_schema,
)
from inspect import signature

AsyncCallable = Callable[..., Awaitable[Any]]

log = logging.getLogger(__name__)


def is_running_in_event_loop() -> bool:
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            return True
        else:
            return False
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
        nest_asyncio.apply()
        assert fn is not None, "fn must be provided"

        self.fn = fn
        self._is_async = iscoroutinefunction(fn)

        self.definition = definition or self._create_fn_definition()
        if self._is_async:
            log.info(f"FunctionTool: {fn} is async: {self._is_async}")

    @property
    def is_async(self) -> bool:
        return self._is_async

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
                    async_tool_1.execute(),
                    sync_tool_1.execute(),
                    async_tool_2.acall(),
                )
                end_time = time.time()
                print(f"run_sync_and_async_mix time: {end_time - start_time}")
                return results

            run_sync_and_async_mix_without_wait()
            asyncio.run(run_sync_and_async_mix())
        """
        if self._is_async:
            log.debug(f"Running async function: {self.fn}")
            if is_running_in_event_loop():
                result = asyncio.create_task(self.acall(*args, **kwargs))
            else:
                result = asyncio.run(self.acall(*args, **kwargs))
        # NOTE: in juptyer notebook, it is always running in event loop
        else:
            log.debug(f"Running sync function: {self.fn}")
            if is_running_in_event_loop():
                log.debug(f"Running sync function in event loop: {self.fn}")
                result = asyncio.to_thread(self.call, *args, **kwargs)
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
        print(results)
        end_time = time.time()
        print(f"run_sync_and_async_mix_without_wait time: {end_time - start_time}")
        return results

    async def run_sync_and_async_mix():
        # both sync and async tool can use execute&to_thread
        # async tool can also use acall without to_thread
        # takes a bit over 2 seconds max(2)
        start_time = time.time()
        results = await asyncio.gather(
            async_tool_1.execute(),
            sync_tool_1.execute(),
            async_tool_2.acall(),
        )
        print(results)
        end_time = time.time()
        print(f"run_sync_and_async_mix time: {end_time - start_time}")
        return results

    print(async_tool_1.execute())

    run_sync_and_async_mix_without_wait()
    asyncio.run(run_sync_and_async_mix())
