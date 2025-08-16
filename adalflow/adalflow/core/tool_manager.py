"""
The ToolManager manages a list of tools, context, and all ways to execute functions.
"""

from typing import (
    List,
    Dict,
    Optional,
    Any,
    Callable,
    Awaitable,
    Union,
    overload,
    Literal,
)
import inspect
import logging
from copy import deepcopy
import asyncio
from adalflow.optim.parameter import Parameter, ParameterType
import nest_asyncio
import warnings

from adalflow.core.container import ComponentList
from adalflow.optim.grad_component import GradComponent
from adalflow.core.component import Component
from adalflow.core.func_tool import FunctionTool
from adalflow.core.types import (
    FunctionDefinition,
    FunctionOutput,
    Function,
    FunctionExpression,
)
from adalflow.utils import printc


from adalflow.core.functional import (
    parse_function_call_expr,
    sandbox_exec,
)

log = logging.getLogger(__name__)


AsyncCallable = Callable[..., Awaitable[Any]]

ToolType = Union[FunctionTool, Callable[..., Any], Awaitable[Callable[..., Any]]]
ToolsType = List[ToolType]


def run_async_in_new_loop(coro):
    """Run async function in a new event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


class CallFunctionTool(Component):
    __doc__ = """Contains other unit gradcomponent such as calling
                a FunctionTool"""

    def __init__(self):
        super().__init__()

    def forward(self, func: Parameter, context: Dict[str, object]):
        return self.bicall(func, context=context)

    def call(self, func: Function, context: Dict[str, object]) -> FunctionOutput:
        return self.bicall(func, context=context)

    def bicall(
        self,
        func: Union[Function, Parameter],
        context: Dict[str, object] = {},
    ):
        if isinstance(func, Parameter):
            # printc(f"context: {context}", color="yellow")
            func_data: Function = func.map_to_successor(self)
            if not isinstance(func_data, Function):
                raise ValueError(f"Error parsing function expression: {func}")
            tool: FunctionTool = context[func_data.name]
            # print(f"tool training: {tool.training}")
            output = tool.forward(*func_data.args, **func_data.kwargs)

            from adalflow.optim.grad_component import fun_to_grad_component

            # this will automatically create the outputparam, and connect output, func to the outputParam
            @fun_to_grad_component()
            def func_to_funcoutput(output, func):
                return output

            # NOTE: for nontrainable function but the overall tool manager is in training mode,
            # we will create a similar func to output edge to handle the feedback backpapagation
            if not isinstance(output, Parameter):
                return func_to_funcoutput.forward(output, func)
            else:
                # reconnect the predecessor for tracing as it is not done in tool.forward
                output.predecessors.add(func)
            return output
        else:
            tool: FunctionTool = context[func.name]
            output = tool.call(*func.args, **func.kwargs)
            return output


class FunctionExperssionToFunction(GradComponent):
    def __init__(self):
        super().__init__(desc="Convert FunctionExpression to Function")

    def call(self, expr: FunctionExpression, context: Dict[str, object]) -> Function:

        assert isinstance(
            expr, FunctionExpression
        ), f"Expected FunctionExpression, got {type(expr)}"

        expr_str = expr.action
        func_name, args, kwargs = parse_function_call_expr(expr_str, context)

        output = Function(
            name=func_name,
            args=args,
            kwargs=kwargs,
            thought=expr.thought,
        )
        return output


# TODO: good to track all the failed function calls
# Tool manager is a task component
class ToolManager(Component):
    __doc__ = r""""Manage a list of tools, context, and all ways to execute functions.


    ToolManager is a task component that does not need its own backward function.

    yaml and json definitions are for quick access to the definitions of the tools.
    If you need more specification, such as using exclude field, you can use the function_definitions.
    """

    def __init__(
        self,
        tools: ToolsType = [],
        additional_context: Optional[
            Dict[str, object]
        ] = {},  # anything besides the tools
    ):
        super().__init__()
        nest_asyncio.apply()  # Apply nest_asyncio to handle nested loops
        processed_tools = [
            (
                FunctionTool(fn=deepcopy(tool))
                if not isinstance(tool, FunctionTool)
                else tool
            )
            for tool in tools
        ]

        self.tools = ComponentList(processed_tools)
        self._context_map = self.create_context_map_from_tools(self.tools)
        self._additional_context = additional_context or {}
        self.context = {**self._context_map, **self._additional_context}
        log.info(
            f"Initialized ToolManager with {len(self.tools)} tools and additional context {self._additional_context}"
        )

    @property
    def context_variables(self) -> Dict[str, object]:
        return self._additional_context

    @staticmethod
    def get_context_index(tool: FunctionTool) -> Dict[str, object]:
        index = tool.definition.func_name
        output = {index: tool}
        return output

    @staticmethod
    def create_context_map_from_tools(tools: List[FunctionTool]) -> Dict[str, object]:
        output: Dict[str, object] = {}
        for tool in tools:
            tool_map = ToolManager.get_context_index(tool)
            for k, v in tool_map.items():
                if k in output:
                    # raise ValueError(f"Duplicate key {k} in the context map.")
                    warnings.warn(f"Duplicate key {k} in the context map.")
                    continue
                output[k] = v
        return output

    @property
    def yaml_definitions(self) -> List[str]:
        output = []
        for tool in self.tools:
            output.append(tool.definition.to_yaml())
        return output

    @property
    def json_definitions(self) -> List[str]:
        output = []
        for tool in self.tools:
            output.append(tool.definition.to_json())
        return output

    @property
    def function_definitions(self) -> List[FunctionDefinition]:
        return [tool.definition for tool in self.tools]

    def parse_func_expr(
        self,
        expr: Union[FunctionExpression, Parameter],
        map_fn: Callable = lambda x: x.data,
    ) -> Union[Function, Parameter]:
        r"""Parse the function call expression."""

        if isinstance(expr, Parameter):
            func = FunctionExperssionToFunction()
            expr.add_successor_map_fn(func, map_fn=map_fn)
            output = func.forward(expr, context=self.context)
            return output
        else:
            try:
                expr_str = expr.action
                func_name, args, kwargs = parse_function_call_expr(
                    expr_str, self.context
                )
                return Function(name=func_name, args=args, kwargs=kwargs)
            except Exception as e:
                log.error(f"Error {e} parsing function call expression: {expr}")
                raise ValueError(f"Error {e} parsing function call expression: {expr}")

    @overload
    def call(
        self, *, expr_or_fun: FunctionExpression, step: Literal["parse"] = "parse"
    ) -> Function: ...

    @overload
    def call(
        self, *, expr_or_fun: FunctionExpression, step: Literal["execute"] = "execute"
    ) -> FunctionOutput: ...

    @overload
    def call(
        self, *, expr_or_fun: Function, step: Literal["execute"] = "parse"
    ) -> Function: ...

    @overload
    def call(
        self, *, expr_or_fun: Function, step: Literal["execute"] = "execute"
    ) -> FunctionOutput: ...

    def call(
        self,
        *,
        expr_or_fun: Union[FunctionExpression, Function],
        step: Literal["execute"] = "execute",
        stream: bool = False,
    ) -> Union[FunctionOutput, Function, Parameter]:
        if not isinstance(expr_or_fun, (Function, FunctionExpression)):
            raise ValueError(
                f"expr_or_fun should be either a Function or FunctionExpression. Got {expr_or_fun}"
            )
        if step == "parse":
            if isinstance(expr_or_fun, Function):
                return expr_or_fun
            return self.parse_func_expr(expr_or_fun)
        elif step == "execute":
            if isinstance(expr_or_fun, Function):
                return self.execute_func(expr_or_fun, stream=stream)
            return self.execute_func_expr(expr_or_fun)
        else:
            raise ValueError(f"step should be either 'parse' or 'execute'. Got {step}")

    def forward(
        self,
        *,
        expr_or_fun: Union[FunctionExpression, Function, Parameter],
        step: Literal["parse", "execute"] = "execute",
        map_fn: Callable = lambda x: x.data,  # how to map the parameter to the needed data
    ) -> Union[FunctionOutput, Function, Parameter]:
        "Run a forward pass on the tool manager such as parsing function expression or executing function."
        if isinstance(expr_or_fun, Parameter):
            expr_or_fun_data = map_fn(expr_or_fun)
            if step == "execute":
                if isinstance(expr_or_fun_data, Function):
                    return self.execute_func(expr_or_fun, map_fn=map_fn)
                else:
                    raise NotImplementedError(
                        "Only Function expressions are supported for now."
                    )
            else:
                if isinstance(expr_or_fun_data, FunctionExpression):
                    output = self.parse_func_expr(expr_or_fun, map_fn=map_fn)
                    return output
                else:
                    raise NotImplementedError(
                        f"Only function call expressions are supported for now. Got {expr_or_fun_data}"
                    )
        else:
            raise ValueError(f"expr_or_fun should be a Parameter. Got {expr_or_fun}")

    def get_function_tool_by_name(self, name: str) -> Optional[FunctionTool]:
        for tool in self.tools:
            if tool.definition.func_name == name:
                return tool
        return None

    def execute_func(
        self,
        func: Union[Function, Parameter],
        map_fn: Callable = lambda x: x.data,
        stream: bool = False,
    ) -> Union[FunctionOutput, Parameter]:
        r"""Execute the function synchronously"""

        if isinstance(func, Parameter):
            try:

                call_func_tool = CallFunctionTool()
                func.add_successor_map_fn(call_func_tool, map_fn=map_fn)
                return call_func_tool.forward(func, context=self.context)

            except Exception as e:
                log.error(f"Error {e} executing function: {func.data}")
                error_msg = f"Error {e} executing function: {func.data}"
                return error_msg

        else:
            try:
                tool: FunctionTool = self.context[func.name]
                log.debug(f"tool: {tool}")

                output = None

                if stream:
                    # add stream = True to the kwargs
                    use_func_kwargs = deepcopy(func.kwargs)
                    use_func_kwargs["stream"] = True
                    output = tool.call(*func.args, **use_func_kwargs)
                else:
                    output = tool.call(*func.args, **func.kwargs)
                    log.debug(f"output: {output}")
                if not isinstance(output, FunctionOutput):
                    raise ValueError(f"Output should be FunctionOutput. Got {output}")
                return output
            except Exception as e:
                log.error(f"Error {e} executing function: {func}")
                raise ValueError(f"Error {e} executing function: {func}")

    # def execute_func(
    #     self,
    #     func: Union[Function, Parameter],
    #     map_fn: Callable = lambda x: x.data,
    #     stream: bool = False,
    # ) -> Union[FunctionOutput, Parameter]:
    #     r"""Execute the function synchronously"""

    #     if isinstance(func, Parameter):
    #         try:

    #             call_func_tool = CallFunctionTool()
    #             func.add_successor_map_fn(call_func_tool, map_fn=map_fn)
    #             return call_func_tool.forward(func, context=self.context)

    #         except Exception as e:
    #             log.error(f"Error {e} executing function: {func.data}")
    #             error_msg = f"Error {e} executing function: {func.data}"
    #             return error_msg

    #     else:
    #         try:
    #             tool: FunctionTool = self.context[func.name]
    #             printc(f"tool: {tool}", color="yellow")
    #             if tool.is_async:
    #                 # Add diagnostic logs
    #                 printc(f"Executing async function: {func.name}")
    #                 result = tool.acall(*func.args, **func.kwargs)
    #                 printc(f"Async result type: {type(result)}")

    #                 # Check if result is an async generator
    #                 import inspect
    #                 return result

    #                 # for streaming
    #                 if inspect.isasyncgen(result):

    #                     # wrap it in FunctionOutput
    #                     result = FunctionOutput(name=func.name, input=func, output=result)
    #                     return result
    #                     # printc("Result is an async generator, collecting results")

    #                     # # We need to handle async generators differently
    #                     # async def collect_async_gen():
    #                     #     items = []
    #                     #     async for item in result:
    #                     #         items.append(item)
    #                     #     return items

    #                     # return run_async_in_new_loop(collect_async_gen())

    #                 # for non-streaming
    #                 else:
    #                     # wrap it in FunctionOutput
    #                     result = FunctionOutput(name=func.name, input=func, output=result)
    #                     return result
    #                     printc("Result is a regular coroutine", color="yellow")
    #                     log.info("Result is a regular coroutine")
    #                     # return run_async_in_new_loop(result)

    #             else:
    #                 printc(f"Executing sync function: {func.name}", color="yellow")
    #                 if stream:
    #                     # add stream = True to the kwargs
    #                     use_func_kwargs = deepcopy(func.kwargs)
    #                     use_func_kwargs["stream"] = True
    #                     output = tool.call(*func.args, **use_func_kwargs)
    #                     return output
    #                 else:
    #                     output = tool.call(*func.args, **func.kwargs)
    #                     printc(f"output: {output}", color="yellow")
    #                     return output
    #         except Exception as e:
    #             log.error(f"Error {e} executing function: {func}")
    #             raise ValueError(f"Error {e} executing function: {func}")

    async def execute_func_async(self, func: Function) -> FunctionOutput:
        r"""Execute the function. If the function is sync, use await to execute it."""
        try:
            log.debug(f"Executing async function: {func.name}")
            tool: FunctionTool = self.context[func.name]
            # await the async call
            try:
                result = tool.acall(*func.args, **func.kwargs)
            except Exception as e:
                error_msg = (
                    f"Error execute_func_async with Error {e} for function: {func}"
                )
                log.error(error_msg)
                raise ValueError(error_msg)

            # it can only be coroutine or function output
            if inspect.iscoroutine(result):
                result = await result
                log.debug(f"result after await: {result}")
            else:
                log.debug("result is not coroutine")

            if not isinstance(result, FunctionOutput):
                error_msg = f"Output should be FunctionOutput. Got {result}"
                log.error(error_msg)
                raise ValueError(error_msg)
            return result
        except Exception as e:
            log.error(f"Error {e} executing function: {func}")
            raise ValueError(f"Error {e} executing function: {func}")


    # async def execute_func_astream(self, func: Function) -> FunctionOutput:
    #     r"""Execute the function. If the function is sync, use await to execute it."""
    #     try:
    #         printc(f"Executing async function: {func.name}", color="yellow")
    #         tool: FunctionTool = self.context[func.name]
    #         # await the async call
    #         try:
    #             result = tool.astream(*func.args, **func.kwargs)
    #         except Exception as e:
    #             error_msg = (
    #                 f"Error execute_func_async with Error {e} for function: {func}"
    #             )
    #             log.error(error_msg)
    #             raise ValueError(error_msg)

    #         # it can only be coroutine or function output
    #         if inspect.iscoroutine(result):
    #             result = await result
    #             printc(f"result after await: {result}", color="yellow")
    #         else:
    #             printc("result is not coroutine", color="yellow")

    #         if not isinstance(result, FunctionOutput):
    #             error_msg = f"Output should be FunctionOutput. Got {result}"
    #             log.error(error_msg)
    #             raise ValueError(error_msg)
    #         return result
    #     except Exception as e:
    #         log.error(f"Error {e} executing function: {func}")
    #         raise ValueError(f"Error {e} executing function: {func}")

    def execute_func_expr(
        self,
        expr: Union[FunctionExpression, Parameter],
        map_fn: Callable = lambda x: x.data,
    ) -> Union[FunctionOutput, Parameter]:
        r"""Execute the function expression. Support both sync and async functions."""

        if isinstance(expr, Parameter):

            func: Parameter = self.parse_func_expr(expr, map_fn=map_fn)
            if not isinstance(func, Parameter):
                raise ValueError(f"Error parsing function expression: {expr}")

            # execute the function
            output: Parameter = self.execute_func(func)
            if not isinstance(output, Parameter):
                raise ValueError(f"Error executing function expression: {expr}")
            output.predecessors.add(expr)
            return output
        else:

            try:
                func: Function = self.parse_func_expr(expr)
                if not isinstance(func, Function):
                    raise ValueError(f"Error parsing function expression: {expr}")

                return self.execute_func(func)
            except Exception as e:
                # NOTE: if the function expression is not a function call, try to execute it as a function expression
                log.error(f"Error {e} executing function expression: {expr}")
                return FunctionOutput(
                    name=expr.action, input=expr, output=None, error=None
                )

    async def execute_func_expr_async(self, expr: FunctionExpression) -> FunctionOutput:
        r"""Execute the function expression. Support both sync and async functions."""
        func: Function = self.parse_func_expr(expr)
        try:
            return await self.execute_func_async(func)
        except Exception as e:
            # NOTE: if the function expression is not a function call, try to execute it as a function expression
            log.error(f"Error {e} executing function expression: {expr}")
            raise ValueError(f"Error {e} executing function expression: {expr}")

    def execute_func_expr_via_sandbox(self, expr: FunctionExpression) -> FunctionOutput:
        r"""Execute the function expression via sandbox. Only support sync functions."""
        func_output = FunctionOutput(
            name=expr.action, input=expr, output=None, error=None
        )
        try:
            action = (
                "output = " + expr.action
                if not expr.action.startswith("output")
                else expr.action
            )
            result = sandbox_exec(action, self.context)
            output = result.get("output", None)
            error = result.get("error", None)
            func_output.output = output
            func_output.error = error

        except Exception as e:
            log.error(f"Error {e} executing function expression: {expr}")
            raise ValueError(f"Error {e} executing function expression: {expr}")

        return func_output

    def execute_func_expr_via_eval(self, expr: FunctionExpression) -> FunctionOutput:
        r"""Execute the function expression via eval. Only support sync functions."""
        try:
            result = eval(expr.action, self.context)
            return FunctionOutput(
                name=expr.action,
                input=expr,
                output=result,
                error=None,
            )
        except Exception as e:
            log.error(f"Error {e} executing function expression: {expr}")
            raise ValueError(f"Error {e} executing function expression: {expr}")

    def _extra_repr(self) -> str:
        s = f"Tools: {self.tools}, Additional Context: {self._additional_context}"
        return s


if __name__ == "__main__":
    # test tool manager
    from adalflow.core.func_tool import FunctionTool
    from adalflow.components.model_client import OpenAIClient
    from adalflow.core.generator import Generator
    from adalflow.optim.parameter import Parameter
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
    # llm_tool.train()
    # output: Parameter = llm_tool("What is 2+2?")
    # output.draw_graph()
    # print(output)

    tool_manager = ToolManager(tools=[llm_tool])
    tool_manager.train()
    expr_or_fun = Parameter(
        name="expr_or_fun",
        data=FunctionExpression(action="llm_as_tool('What is 2+2?')"),
        eval_input="What is 2+2?",
        param_type=ParameterType.INPUT,
    )
    output: Parameter = tool_manager(expr_or_fun=expr_or_fun, step="parse")
    print(output)
    print(output.predecessors)
    assert len(output.predecessors) == 1
    # output = tool_manager(output, step="execute")
    # print(output)
    # output.draw_graph()

    # expr_or_fun = FunctionExpression(action="llm_as_tool('What is 2+2?')")

    # tool_manager.eval()
    # output = tool_manager(expr_or_fun=expr_or_fun, step="execute")
    # print(output)
