"""
Tool is LLM's extended capability which is one of the core design pattern of Agent. All tools can be wrapped in a FunctionTool class.
This helps to standardize the tool interface and metadata to communicate with the Agent.
"""

from typing import Any, Optional, Callable, Awaitable, Union
from inspect import iscoroutinefunction, ismethod, isfunction
import inspect
import logging
import asyncio
import nest_asyncio


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


FunctionType = Union[Callable[..., Any], Awaitable[Callable[..., Any]]]


# TODO: improve the support for async functions, similarly a component might be used as a tool
class FunctionTool(Component):
    __doc__ = r"""Describing and Parsing(to LLM) and executing a function.

    Supports both normal callable functions and methods(__call__) of a component.
    When component is used, we support both the training and eval mode.

    Note:

        When the eval mode, it outputs FunctionOutput, and when the training mode, it outputs Parameter with data as FunctionOutput.

    Args:
        fn (Callable): The function to be executed.
        component (Component, optional): The component that owns the function. Defaults to None.
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

    def __init__(
        self,
        fn: FunctionType,
        component: Optional[Component] = None,
        definition: Optional[FunctionDefinition] = None,
    ):
        super().__init__(
            name="FunctionTool", desc="A component calls and executes a function."
        )
        nest_asyncio.apply()
        assert fn is not None, "fn must be provided"

        # self.fn = fn  # it can be a function or component
        self.component = component  # pass it here to control the training mode
        if isinstance(fn, Component):
            self.fn = fn.__call__
        else:
            self.fn = fn
        self._is_async = iscoroutinefunction(fn)
        if isinstance(fn, Component):
            self.definition = (
                definition or self._create_fn_definition_for_grad_component(fn)
            )
        else:
            self.definition = definition or self._create_fn_definition()
        if self._is_async:
            log.info(f"FunctionTool: {fn} is async: {self._is_async}")

    @property
    def is_async(self) -> bool:
        return self._is_async

    def _create_fn_definition_for_grad_component(
        self, fn: FunGradComponent
    ) -> FunctionDefinition:
        name = fn.fun_name
        docstring = fn.doc_string
        signature_str = str(signature(fn.fun))
        instance = None
        return FunctionDefinition(
            func_name=name,
            func_desc=(
                f"{name}{signature_str}\nDocstring:{docstring}"
                if isinstance(docstring, str)
                else f"{name}{signature_str}\nDocstring:{docstring.data}"
            ),
            func_parameters=get_fun_schema(name, fn.fun),
            class_instance=instance,
        )

    def _create_fn_definition(self) -> FunctionDefinition:

        name = self.fn.__name__
        docstring = self.fn.__doc__
        signature_str = str(signature(self.fn))

        # Get the class that owns the method, if applicable
        cls_name = None
        instance = None
        if ismethod(self.fn):  # Check if it’s a bound method
            instance = self.fn.__self__
            instance = find_instance_name_from_self(instance)
            if name == "__call__" and not instance:
                raise ValueError(
                    "Please provide a name for the instance in the calling context"
                )
            cls_name = self.fn.__self__.__class__.__name__
        elif isfunction(self.fn):  # Unbound method
            cls_name = self.fn.__qualname__.split(".")[0]

        # Build the description
        description = f"{name}{signature_str}\n"
        if cls_name:
            description += f"Belongs to class: {cls_name}\n"
        if docstring:
            description += f"Docstring: {docstring}\n"

        # Get function parameters schema
        fn_parameters = get_fun_schema(name, self.fn)

        return FunctionDefinition(
            func_name=name,
            func_desc=description,
            func_parameters=fn_parameters,
            class_instance=instance,
        )

    def forward(self, *args, **kwargs) -> Parameter:
        r"""Forward the function tool."""
        return self.bicall(*args, **kwargs)

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
        return self.bicall(*args, **kwargs)
        # if self._is_async:
        #     raise ValueError("FunctionTool is asynchronous, use acall instead")
        # output, error = None, None
        # try:
        #     output = self.fn(*args, **kwargs)
        # except Exception as e:
        #     log.error(f"Error at calling {self.fn}: {e}")
        #     # raise ValueError(f"Error: {e}")
        #     error = str(e)
        # return FunctionOutput(
        #     name=self.definition.func_name,
        #     # raw_input={"args": args, "kwargs": kwargs},
        #     input=Function(name=self.definition.func_name, args=args, kwargs=kwargs),
        #     output=output,
        #     error=error,
        # )

    def bicall(self, *args: Any, **kwargs: Any) -> Union[FunctionOutput, Parameter]:
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

        # NOTE: special case:
        # self.fn can have both train and eval mode or untrainable as a function.
        try:
            # printc(f"args: {args}, kwargs: {kwargs}, fn: {self.fn}", color="yellow")
            output = self.fn(*args, **kwargs)
            # printc(f"output 1: {output}", color="yellow")
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
        # printc(f"output: {output}", color="yellow")
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

    # def __call__(self, *args, **kwargs) -> FunctionOutput:
    #     r"""Execute the function synchronously or asynchronously based on the function type."""
    #     return self.execute(*args, **kwargs)

    def _extra_repr(self) -> str:
        s = f"fn: {self.fn}, async: {self._is_async}, definition: {self.definition}"
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
