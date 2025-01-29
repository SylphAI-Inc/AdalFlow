"""
The ToolManager manages a list of tools, context, and all ways to execute functions.
"""

from typing import List, Dict, Optional, Any, Callable, Awaitable, Union
import logging
from copy import deepcopy
import asyncio
import nest_asyncio

from adalflow.core import Component
from adalflow.core.func_tool import FunctionTool
from adalflow.core.types import (
    FunctionDefinition,
    FunctionOutput,
    Function,
    FunctionExpression,
)

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


# TODO: good to track all the failed function calls
class ToolManager(Component):
    __doc__ = r""""Manage a list of tools, context, and all ways to execute functions.

    yaml and json definitions are for quick access to the definitions of the tools.
    If you need more specification, such as using exclude field, you can use the function_definitions.
    Args:


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
        # super(LocalDB, self).__init__()
        self.tools = [
            (
                FunctionTool(fn=deepcopy(tool))
                if not isinstance(tool, FunctionTool)
                else deepcopy(tool)
            )
            for tool in tools
        ]
        self._context_map = {tool.definition.func_name: tool for tool in self.tools}
        self._additional_context = additional_context or {}
        self.context = {**self._context_map, **self._additional_context}
        log.info(
            f"Initialized ToolManager with {len(self.tools)} tools and additional context {self._additional_context}"
        )

    @property
    def yaml_definitions(self) -> List[str]:
        return [tool.definition.to_yaml() for tool in self.tools]

    @property
    def json_definitions(self) -> List[str]:
        return [tool.definition.to_json() for tool in self.tools]

    @property
    def function_definitions(self) -> List[FunctionDefinition]:
        return [tool.definition for tool in self.tools]

    def parse_func_expr(self, expr: FunctionExpression) -> Function:
        r"""Parse the function call expression."""
        try:
            expr_str = expr.action
            func_name, args, kwargs = parse_function_call_expr(expr_str, self.context)
            return Function(name=func_name, args=args, kwargs=kwargs)
        except Exception as e:
            log.error(f"Error {e} parsing function call expression: {expr_str}")
            raise ValueError(f"Error {e} parsing function call expression: {expr_str}")

    def execute_func(self, func: Function) -> FunctionOutput:
        r"""Execute the function. If the function is async, use asyncio.run to execute it."""
        try:
            tool: FunctionTool = self.context[func.name]
            if tool.is_async:
                log.debug("Running async function in new loop")
                return run_async_in_new_loop(tool.acall(*func.args, **func.kwargs))
            else:
                return tool.call(*func.args, **func.kwargs)
        except Exception as e:
            log.error(f"Error {e} executing function: {func}")
            raise ValueError(f"Error {e} executing function: {func}")

    async def execute_func_async(self, func: Function) -> FunctionOutput:
        r"""Execute the function. If the function is sync, use await to execute it."""
        try:
            tool: FunctionTool = self.context[func.name]
            if tool.is_async:
                return await tool.acall(*func.args, **func.kwargs)
            else:
                return asyncio.to_thread(self.call, *func.args, **func.kwargs)
        except Exception as e:
            log.error(f"Error {e} executing function: {func}")
            raise ValueError(f"Error {e} executing function: {func}")

    def execute_func_expr(self, expr: FunctionExpression) -> FunctionOutput:
        r"""Execute the function expression. Support both sync and async functions."""
        func: Function = self.parse_func_expr(expr)
        try:

            return self.execute_func(func)
        except Exception as e:
            # NOTE: if the function expression is not a function call, try to execute it as a function expression
            log.error(f"Error {e} executing function expression: {expr}")
            raise ValueError(f"Error {e} executing function expression: {expr}")

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
