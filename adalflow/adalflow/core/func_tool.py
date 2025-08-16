"""
Tool is LLM's extended capability which is one of the core design pattern of Agent. All tools can be wrapped in a FunctionTool class.
This helps to standardize the tool interface and metadata to communicate with the Agent.
"""

from typing import Any, Optional, Callable, Awaitable, Union
from inspect import ismethod
import inspect
import logging
import asyncio
import nest_asyncio
from enum import Enum, auto


from adalflow.core.types import (
    FunctionDefinition,
    FunctionOutput,
    Function,
)
from adalflow.core import Component
from adalflow.optim.parameter import Parameter
from adalflow.optim.grad_component import FunGradComponent
from adalflow.core.functional import (
    get_fun_schema,
)
from adalflow.utils import printc
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


def find_instance_name_from_self(instance):
    """
    Attempt to find the variable name of the instance in the calling context.

    :param instance: The instance to find the name for.
    :return: The variable name of the instance, if found; otherwise, None.
    """
    # Inspect the calling stack frame
    frame = inspect.stack()[2].frame
    for var_name, var_obj in frame.f_locals.items():
        if var_obj is instance:
            return var_name
    return None


# Specific function types supported by FunctionTool:
# - Regular functions (sync/async)
# - Generator functions (sync/async)
# - Bound methods (class methods)
# - FunGradComponent instances (trainable components)


class FunctionType(Enum):
    """Enumeration of the 4 core function types supported by FunctionTool."""

    SYNC = auto()  # Regular sync function: def func(): return value
    ASYNC = auto()  # Async function: async def func(): return value
    SYNC_GENERATOR = auto()  # Sync generator: def func(): yield value
    ASYNC_GENERATOR = auto()  # Async generator: async def func(): yield value


# TODO: Add a wrapper to add **kwargs at the end of each function
class FunctionTool(Component):
    __doc__ = r"""Describing and Parsing(to LLM) and executing a function.

    Supports both normal callable functions and class methods.
    When component is used, we support both the training and eval mode.

    Note:

        When the eval mode, it outputs FunctionOutput, and when the training mode, it outputs Parameter with data as FunctionOutput.

    Args:
        fn (Callable): The function to be executed.
        definition (FunctionDefinition, optional): The definition of the function. Defaults to None.


    Function be used by LLM as a tool to achieve a specific task.

    What function can you pass as a tool?
    1. Any unbound function you wrote outside of a class.
    2. Any class method you wrote in your component. It can call `self` and other methods inside of your component.
    3. When the function is using a trainable component, and you can directly use the component's method as a tool or wrap it in a function. But you need to make sure to pass the component to the tool.

    Here are some examples:

    .. code-block:: python

        from adalflow.core.func_tool import FunctionTool
        class AgenticRAG(Component):
            def __init__(self, ...):
                super().__init__()
                self.retriever = Retriever()
                self.llm = Generator()

                def retriever_as_tool(input: str) -> str:
                    r"Used as a retriever tool."
                    return self.retriever(input)

                tools = [FunctionTool(retriever_as_tool, component=self.retriever),
                            FunctionTool(self.llm.__call__, component=self.llm)]
                # if you have trainable component, this will ensure it can be trained together with your whole task pipeline
                # if you dont want to train them and simply treating them as a tool, you can call like this
                # tools = [FunctionTool(retriever_as_tool), FunctionTool(self.llm.__call__, component=self.llm)]

    Features:

    - Supports both synchronous and asynchronous functions via `call` and `acall`.
    - Creates a `FunctionDefinition` from the function using `get_fun_schema`.
    - Executes the function with arguments.
        - Parses the function call expression (`FunctionExpression`) into `Function` (name, args, kwargs).
        - Executes the function using one of the following methods:
            - Via `call` with args and kwargs.
            - Via `eval`, without any context or sandboxing.
            - Via sandboxed execution directly using `sandbox_exec`.

    A FunctionTool allows other GradComponent(as a tool) to pass through correctly.
    """

    # key attributes:
    fn: Callable
    definition: FunctionDefinition
    function_type: FunctionType

    # it inherits the training attribute from Component
    def __init__(
        self,
        fn: Union[Callable, FunGradComponent],
        definition: Optional[FunctionDefinition] = None,
        require_approval: bool = False,
        pre_execute_callback: Optional[Callable] = None,
    ):
        super().__init__(
            name="FunctionTool", desc="A component calls and executes a function."
        )
        nest_asyncio.apply()
        assert fn is not None, "fn must be provided"

        # TODO: support FunGradComponent later.

        self.fn = fn
        self.require_approval = require_approval
        self.pre_execute_callback = pre_execute_callback # executed before the function is called, often useful for generating confirmation logics to the user
        self.function_type = self.detect_function_type(fn)
        self._is_async = self.function_type in [
            FunctionType.ASYNC,
            FunctionType.ASYNC_GENERATOR,
        ]
        self.class_instance = self._autodetect_class_instance(fn)
        if isinstance(fn, FunGradComponent):
            print(f"FunctionTool: {fn} is a component")
            self.definition = (
                definition or self._create_fn_definition_for_grad_component(fn)
            )
        else:
            self.definition = definition or self._create_fn_definition()
        if self._is_async:
            log.info(f"FunctionTool: {fn} is async: {self._is_async}")

    @classmethod
    def detect_function_type(cls, fn: Callable) -> FunctionType:
        """
        Detect the function type of a given callable.

        Args:
            fn: The callable to analyze

        Returns:
            FunctionType: The detected function type

        Raises:
            ValueError: If the function type cannot be determined or is not supported
        """
        if fn is None:
            raise ValueError("Function cannot be None")

        # Check for async generator functions
        if inspect.isasyncgenfunction(fn):
            return FunctionType.ASYNC_GENERATOR

        # Check for sync generator functions
        if inspect.isgeneratorfunction(fn):
            return FunctionType.SYNC_GENERATOR

        # Check for async functions (coroutines)
        if inspect.iscoroutinefunction(fn):
            return FunctionType.ASYNC

        # Check for regular functions
        if inspect.isfunction(fn) or inspect.ismethod(fn):
            return FunctionType.SYNC

        # Check for callable objects (like classes with __call__)
        if callable(fn):
            # For callable objects, we need to check their __call__ method
            if hasattr(fn, "__call__"):
                call_method = fn.__call__
                if inspect.ismethod(call_method):
                    # It's a bound method, check the underlying function
                    if inspect.isasyncgenfunction(call_method.__func__):
                        return FunctionType.ASYNC_GENERATOR
                    elif inspect.isgeneratorfunction(call_method.__func__):
                        return FunctionType.SYNC_GENERATOR
                    elif inspect.iscoroutinefunction(call_method.__func__):
                        return FunctionType.ASYNC
                    else:
                        return FunctionType.SYNC
                else:
                    # It's a function, check directly
                    if inspect.isasyncgenfunction(call_method):
                        return FunctionType.ASYNC_GENERATOR
                    elif inspect.isgeneratorfunction(call_method):
                        return FunctionType.SYNC_GENERATOR
                    elif inspect.iscoroutinefunction(call_method):
                        return FunctionType.ASYNC
                    else:
                        return FunctionType.SYNC

        raise ValueError(f"Cannot determine function type for {fn}")

    def _create_fn_definition_for_grad_component(
        self, fn: FunGradComponent
    ) -> FunctionDefinition:
        name = fn.fun_name
        docstring = fn.doc_string
        signature_str = str(signature(fn.fun))
        cls_name = None
        if ismethod(fn.fun):
            cls_name = fn.fun.__self__.__class__.__name__

        name = cls_name + "_" + name if cls_name else name
        return FunctionDefinition(
            func_name=name,
            func_desc=(
                f"{name}{signature_str}\nDocstring:{docstring}"
                if isinstance(docstring, str)
                else f"{name}{signature_str}\nDocstring:{docstring.data}"
            ),
            func_parameters=get_fun_schema(name, fn.fun),
        )

    def _autodetect_class_instance(self, fn: Callable) -> Optional[Any]:
        if ismethod(fn):
            return fn.__self__
        return None

    def _create_fn_definition(self) -> FunctionDefinition:

        name = self.fn.__name__
        docstring = self.fn.__doc__
        signature_str = str(signature(self.fn))

        # Get the class that owns the method, if applicable
        cls_name = None
        if ismethod(self.fn):  # Check if it's a bound method
            cls_name = self.fn.__self__.__class__.__name__

        # Build the description
        description = f"{name}{signature_str}\n"
        if cls_name:
            description += f"Belongs to class: {cls_name}\n"
        if docstring:
            description += f"Docstring: {docstring}\n"

        # Get function parameters schema
        fn_parameters = get_fun_schema(name, self.fn)

        name = cls_name + "_" + name if cls_name else name
        # create a unique identifier as the class method name

        return FunctionDefinition(
            func_name=name,
            func_desc=description,
            func_parameters=fn_parameters,
        )

    def forward(self, *args, **kwargs) -> Parameter:
        r"""Forward the function tool."""
        return self.bicall(*args, **kwargs)

    def _call_sync(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Call a sync function."""
        if self.function_type == FunctionType.SYNC:
            return fn(*args, **kwargs)
        elif self.function_type == FunctionType.ASYNC:
            if is_running_in_event_loop():
                loop = asyncio.get_running_loop()
                task = loop.create_task(fn(*args, **kwargs))
                return asyncio.run_coroutine_threadsafe(task, loop).result()
            else:
                return asyncio.run(fn(*args, **kwargs))
        elif self.function_type == FunctionType.SYNC_GENERATOR:
            return fn(*args, **kwargs)
            # output = []
            # sync_gen = fn(*args, **kwargs)
            # for event in sync_gen:
            #     output.append(event)
            # return output

        elif self.function_type == FunctionType.ASYNC_GENERATOR:
            # For async generators in sync context, just return the generator object
            # The runner will handle the collection of events
            return fn(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported function type: {self.function_type}")

    def call(self, *args: Any, **kwargs: Any) -> FunctionOutput:
        """
        Execute the function synchronously, supporting all function types.

        This method provides a unified sync interface for all function types:
        - SYNC: Calls the function directly
        - ASYNC: Runs the coroutine in a new event loop (blocks until complete)
        - SYNC_GENERATOR: Returns the generator object
        - ASYNC_GENERATOR: Runs the async generator and collects all values into a list

        Warning: For async functions, this will block the current thread until completion.
        For better performance with async functions, consider using acall() instead.

        Example:
            import asyncio

            async def async_func():
                await asyncio.sleep(1)
                return "async result"

            def sync_func():
                return "sync result"

            tool1 = FunctionTool(async_func)
            tool2 = FunctionTool(sync_func)

            # Both work synchronously
            result1 = tool1.call()  # Blocks for 1 second
            result2 = tool2.call()  # Returns immediately
        """
        output, error = None, None

        try:
            output = self._call_sync(self.fn, *args, **kwargs)

        except Exception as e:
            log.error(f"Error at calling {self.fn}: {e}")
            error = f"Error at calling {self.fn}: {e}"

        # Handle Parameter output (training mode)
        if isinstance(output, Parameter):
            if not self.training:
                raise ValueError(
                    f"FunctionTool {self.definition.func_name} is in eval mode, but the output is Parameter"
                )
            output.data = FunctionOutput(
                name=self.definition.func_name,
                input=Function(
                    name=self.definition.func_name, args=args, kwargs=kwargs
                ),
                output=output.data,
                error=error,
            )
            return output

        # Create FunctionOutput
        function_output = FunctionOutput(
            name=self.definition.func_name,
            input=Function(name=self.definition.func_name, args=args, kwargs=kwargs),
            output=output,
            error=error,
        )

        log.debug(f"call output: {function_output}")
        return function_output

    def bicall(self, *args: Any, **kwargs: Any) -> Union[FunctionOutput, Parameter]:
        r"""This should only be used in training, where a fun is required to be a FunGradComponent
        where the output from function execution is a Parameter.
        """
        if self._is_async:
            raise ValueError("FunctionTool is asynchronous, use acall instead")
        output, error = None, None

        # NOTE: special case:
        # self.fn can have both train and eval mode or untrainable as a function.
        try:
            log.debug(f"bicall args: {args}, kwargs: {kwargs}, fn: {self.fn}")
            # TODO: might to support more types of functions
            output = self.fn(*args, **kwargs)
        except Exception as e:
            log.error(f"Error at calling {self.fn}: {e}")
            error = f"Error at calling {self.fn}: {e}"

        if isinstance(output, Parameter):
            if not self.training:
                raise ValueError(
                    f"FunctionTool {self.definition.func_name} is in eval mode, but the output is Parameter"
                )
            output.data = FunctionOutput(
                name=self.definition.func_name,
                # raw_input={"args": args, "kwargs": kwargs},
                input=Function(
                    name=self.definition.func_name, args=args, kwargs=kwargs
                ),
                output=output.data,
                error=error,
            )
            return output

        output = FunctionOutput(
            name=self.definition.func_name,
            input=Function(name=self.definition.func_name, args=args, kwargs=kwargs),
            output=output,
            error=error,
        )
        log.debug(f"function output: {output}")
        return output

    async def acall(self, *args, **kwargs) -> Union[FunctionOutput, Parameter]:
        """
        Async call the function. Handles all function types appropriately.

        For different function types:
        - SYNC: Returns FunctionOutput with the result
        - ASYNC: Awaits the coroutine and returns FunctionOutput with the result
        - SYNC_GENERATOR: Returns FunctionOutput with the generator object
        - ASYNC_GENERATOR: Returns FunctionOutput with the async generator object

        Note: For generators, users need to iterate over the generator themselves.
        """
        output, error = None, None
        log.debug(f"output arguments: {args}, {kwargs}")

        try:
            if self.function_type == FunctionType.SYNC:
                # Sync function - call directly
                output = self.fn(*args, **kwargs)
                log.debug(f"output in synchronous function call: {output}")

            elif self.function_type == FunctionType.ASYNC:
                # Async function - await the coroutine
                output = await self.fn(*args, **kwargs)

            elif self.function_type == FunctionType.SYNC_GENERATOR:
                # Sync generator - return the generator object
                output = self.fn(*args, **kwargs)

            elif self.function_type == FunctionType.ASYNC_GENERATOR:
                # Async generator - return the async generator object
                output = self.fn(*args, **kwargs)

            else:
                raise ValueError(f"Unsupported function type: {self.function_type}")

        except Exception as e:
            log.error(f"Error at calling {self.fn}: {e}")
            error = f"Error at calling {self.fn}: {e}"

        # Handle Parameter output (training mode)
        if isinstance(output, Parameter):
            if not self.training:
                raise ValueError(
                    f"FunctionTool {self.definition.func_name} is in eval mode, but the output is Parameter"
                )
            output.data = FunctionOutput(
                name=self.definition.func_name,
                input=Function(
                    name=self.definition.func_name, args=args, kwargs=kwargs
                ),
                output=output.data,
                error=error,
            )
            return output

        function_output = FunctionOutput(
            name=self.definition.func_name,
            input=Function(name=self.definition.func_name, args=args, kwargs=kwargs),
            output=output,
            error=error,
        )

        return function_output

    # async def acall(self, *args, **kwargs) -> Union[FunctionOutput, Parameter]:
    #     """
    #     Async call the function with automatic event collection for generators.

    #     For different function types:
    #     - SYNC: Returns FunctionOutput with the result
    #     - ASYNC: Awaits the coroutine and returns FunctionOutput with the result
    #     - SYNC_GENERATOR: Collects all values from the generator into a list
    #     - ASYNC_GENERATOR: Collects all values from the async generator into a list

    #     This method automatically collects all yielded values from generators into a list,
    #     similar to how the sync call method handles async generators.
    #     """
    #     output, error = None, None

    #     try:
    #         if self.function_type == FunctionType.SYNC:
    #             # Sync function - call directly
    #             output = self.fn(*args, **kwargs)

    #         elif self.function_type == FunctionType.ASYNC:
    #             # Async function - await the coroutine
    #             output = await self.fn(*args, **kwargs)

    #         elif self.function_type == FunctionType.SYNC_GENERATOR:
    #             # Sync generator - collect all values into a list
    #             generator = self.fn(*args, **kwargs)
    #             output = []
    #             for item in generator:
    #                 output.append(item)

    #         elif self.function_type == FunctionType.ASYNC_GENERATOR:
    #             # Async generator - collect all values into a list
    #             async_generator = self.fn(*args, **kwargs)
    #             output = []
    #             async for item in async_generator:
    #                 output.append(item)

    #         else:
    #             raise ValueError(f"Unsupported function type: {self.function_type}")

    #     except Exception as e:
    #         log.error(f"Error at calling {self.fn}: {e}")
    #         error = f"Error at calling {self.fn}: {e}"

    #     # Handle Parameter output (training mode)
    #     if isinstance(output, Parameter):
    #         if not self.training:
    #             raise ValueError(
    #                 f"FunctionTool {self.definition.func_name} is in eval mode, but the output is Parameter"
    #             )
    #         output.data = FunctionOutput(
    #             name=self.definition.func_name,
    #             input=Function(
    #                 name=self.definition.func_name, args=args, kwargs=kwargs
    #             ),
    #             output=output.data,
    #             error=error,
    #         )
    #         return output

    #     function_output = FunctionOutput(
    #         name=self.definition.func_name,
    #         input=Function(name=self.definition.func_name, args=args, kwargs=kwargs),
    #         output=output,
    #         error=error,
    #     )

    #     return function_output

    # def execute(self, *args, **kwargs) -> FunctionOutput:
    #     r"""Execute the function synchronously or asynchronously based on the function type.

    #     No matter of the function type, you can run the function using both asyncio and without asyncio.

    #     Use it with caution as it might block the event loop.

    #     Example:

    #     .. code-block:: python

    #         import asyncio
    #         import time

    #         async def async_function_1():
    #             await asyncio.sleep(1)
    #             return "Function 1 completed"

    #         def sync_function_1():
    #             time.sleep(1)
    #             return "Function 1 completed"

    #         async def async_function_2():
    #             await asyncio.sleep(2)
    #             return "Function 2 completed"

    #         def sync_function_2():
    #             time.sleep(2)
    #             return "Function 2 completed"

    #         async_tool_1 = FunctionTool(async_function_1)
    #         sync_tool_1 = FunctionTool(sync_function_2)
    #         async_tool_2 = FunctionTool(async_function_2)
    #         sync_tool_2 = FunctionTool(sync_function_2)

    #         def run_sync_and_async_mix_without_wait():
    #             # both sync and async tool can use execute
    #             # sync tool can also use call
    #             # takes 5 seconds (1+1+2) + overhead
    #             start_time = time.time()
    #             results = [
    #                 async_tool_1.execute(),
    #                 sync_tool_1.execute(),
    #                 sync_tool_2.call(),
    #             ]
    #             end_time = time.time()
    #             print(f"run_sync_and_async_mix_without_wait time: {end_time - start_time}")
    #             return results

    #         async def run_sync_and_async_mix():
    #             # both sync and async tool can use execute&to_thread
    #             # async tool can also use acall without to_thread
    #             # takes a bit over 2 seconds max(2)
    #             start_time = time.time()
    #             results = await asyncio.gather(
    #                 async_tool_1.execute(),
    #                 sync_tool_1.execute(),
    #                 async_tool_2.acall(),
    #             )
    #             end_time = time.time()
    #             print(f"run_sync_and_async_mix time: {end_time - start_time}")
    #             return results

    #         run_sync_and_async_mix_without_wait()
    #         asyncio.run(run_sync_and_async_mix())
    #     """
    #     if self._is_async:
    #         log.debug(f"Running async function: {self.fn}")
    #         if is_running_in_event_loop():
    #             result = asyncio.create_task(self.acall(*args, **kwargs))
    #         else:
    #             result = asyncio.run(self.acall(*args, **kwargs))
    #     # NOTE: in juptyer notebook, it is always running in event loop
    #     else:
    #         log.debug(f"Running sync function: {self.fn}")
    #         if is_running_in_event_loop():
    #             log.debug(f"Running sync function in event loop: {self.fn}")
    #             result = asyncio.to_thread(self.call, *args, **kwargs)
    #         else:
    #             result = self.call(*args, **kwargs)

    #     return result

    # def __call__(self, *args, **kwargs) -> FunctionOutput:
    #     r"""Execute the function synchronously or asynchronously based on the function type."""
    #     return self.execute(*args, **kwargs)

    def _extra_repr(self) -> str:
        s = f"fn: {self.fn}, type: {self.function_type}, definition: {self.definition}"
        if self.class_instance is not None:
            s += f", class_instance: {self.class_instance}"
        return s


if __name__ == "__main__":

    # import asyncio
    # import time

    # async def async_function_1():
    #     await asyncio.sleep(1)
    #     return "Function 1 completed"

    # def sync_function_1():
    #     time.sleep(1)
    #     return "Function 1 completed"

    # async def async_function_2():
    #     await asyncio.sleep(2)
    #     return "Function 2 completed"

    # def sync_function_2():
    #     time.sleep(2)
    #     return "Function 2 completed"

    # async_tool_1 = FunctionTool(async_function_1)
    # sync_tool_1 = FunctionTool(sync_function_2)
    # async_tool_2 = FunctionTool(async_function_2)
    # sync_tool_2 = FunctionTool(sync_function_2)

    # def run_sync_and_async_mix_without_wait():
    #     # both sync and async tool can use execute
    #     # sync tool can also use call
    #     # takes 5 seconds (1+1+2) + overhead
    #     start_time = time.time()
    #     results = [
    #         async_tool_1.execute(),
    #         sync_tool_1.execute(),
    #         sync_tool_2.call(),
    #     ]
    #     print(results)
    #     end_time = time.time()
    #     print(f"run_sync_and_async_mix_without_wait time: {end_time - start_time}")
    #     return results

    # async def run_sync_and_async_mix():
    #     # both sync and async tool can use execute&to_thread
    #     # async tool can also use acall without to_thread
    #     # takes a bit over 2 seconds max(2)
    #     start_time = time.time()
    #     results = await asyncio.gather(
    #         async_tool_1.execute(),
    #         sync_tool_1.execute(),
    #         async_tool_2.acall(),
    #     )
    #     print(results)
    #     end_time = time.time()
    #     print(f"run_sync_and_async_mix time: {end_time - start_time}")
    #     return results

    # print(async_tool_1.execute())

    # run_sync_and_async_mix_without_wait()
    # asyncio.run(run_sync_and_async_mix())

    from adalflow.components.model_client import OpenAIClient
    from adalflow.core.generator import Generator
    from adalflow.optim.parameter import Parameter
    from adalflow.core.types import GeneratorOutput
    from adalflow.utils import setup_env, printc

    setup_env()

    llm = Generator(
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-3.5-turbo"},
    )
    # llm.train()

    def llm_as_tool(input: str, id: Optional[str] = None) -> str:
        """Used as a calculator tool."""
        printc(f"llm_as_tool: {input}", color="yellow")

        return llm(prompt_kwargs={"input_str": input}, id=id)

    llm_tool = FunctionTool(llm_as_tool, component=llm)
    llm_tool.train()
    output: Parameter = llm_tool("What is 2+2?")
    output.draw_graph()
    print(output)
    llm_tool.eval()
    output: FunctionTool = llm_tool("What is 2+2?")
    print(output)
    assert isinstance(output, FunctionOutput)
    assert isinstance(output.output, GeneratorOutput)

    # grad component

    from adalflow.optim.grad_component import fun_to_grad_component
    from adalflow.optim.parameter import ParameterType

    @fun_to_grad_component(
        desc="Finish",
        doc_string=Parameter(
            data="Finish the task with verbatim short factoid responses from retrieved context.",
            param_type=ParameterType.PROMPT,
            requires_opt=True,
            role_desc="Instruct how the agent creates the final answer from the step history.",
            name="doc_string",
        ),
    )
    def finish(answer: str, **kwargs) -> str:
        # """Finish the task with verbatim short factoid responses from retrieved context."""
        # printc(f"finish: {answer}", color="yellow")
        return answer

    finish_tool = FunctionTool(fn=finish, component=finish)

    definition = finish_tool.definition
    print(definition)
    # call function
    finish_tool.train()
    output: Parameter = finish_tool(
        "Finish the task with verbatim short factoid responses from retrieved context."
    )
    print(output)
