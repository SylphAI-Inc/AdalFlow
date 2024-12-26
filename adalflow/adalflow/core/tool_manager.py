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
import logging
from copy import deepcopy
import asyncio
from adalflow.optim.parameter import Parameter, ParameterType
import nest_asyncio

from adalflow.core.container import ComponentList
from adalflow.optim.grad_component import GradComponent
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


# TODO: good to track all the failed function calls
# Tool manager is a task component
class ToolManager(GradComponent):
    __doc__ = r""""Manage a list of tools, context, and all ways to execute functions.

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
        tools = [
            (
                FunctionTool(fn=deepcopy(tool))
                if not isinstance(tool, FunctionTool)
                else tool
            )
            for tool in tools
        ]
        self.tools = ComponentList(tools)
        self._context_map = self.create_context_map_from_tools(self.tools)
        self._additional_context = additional_context or {}
        self.context = {**self._context_map, **self._additional_context}
        log.info(
            f"Initialized ToolManager with {len(self.tools)} tools and additional context {self._additional_context}"
        )

    @staticmethod
    def get_context_index(tool: FunctionTool) -> Dict[str, object]:
        index = tool.definition.func_name
        if tool.definition.class_instance:
            index = f"{tool.definition.class_instance}.{index}"
        output = {index: tool}
        if tool.definition.func_name == "__call__":
            # add another index of directly using the classinstance
            output[f"{tool.definition.class_instance}"] = tool
        return output

    @staticmethod
    def create_context_map_from_tools(tools: List[FunctionTool]) -> Dict[str, object]:
        output: Dict[str, object] = {}
        for tool in tools:
            tool_map = ToolManager.get_context_index(tool)
            for k, v in tool_map.items():
                if k in output:
                    raise ValueError(f"Duplicate key {k} in the context map.")
                output[k] = v
        return output

    @property
    def yaml_definitions(self) -> List[str]:
        output = []
        for tool in self.tools:
            if not tool.definition.class_instance:
                output.append(tool.definition.to_yaml(exclude=["class_instance"]))
            else:
                output.append(tool.definition.to_yaml())
            output.append(tool.definition.to_yaml(exclude=["class_instance"]))
        return output

    @property
    def json_definitions(self) -> List[str]:
        output = []
        for tool in self.tools:
            if not tool.definition.class_instance:
                output.append(tool.definition.to_json(exclude=["class_instance"]))
            else:
                output.append(tool.definition.to_json())
            output.append(tool.definition.to_json(exclude=["class_instance"]))
        return output

    @property
    def function_definitions(self) -> List[FunctionDefinition]:
        return [tool.definition for tool in self.tools]

    def parse_func_expr(
        self, expr: Union[FunctionExpression, Parameter]
    ) -> Union[Function, Parameter]:
        r"""Parse the function call expression."""

        if isinstance(expr, Parameter):
            try:

                class FunctionExperssionToFunction(GradComponent):
                    def __init__(self):
                        super().__init__()

                    def call(
                        self, expr: FunctionExpression, context: Dict[str, object]
                    ):
                        print("DummpyGradComponent call")
                        print(expr)

                        expr_str = expr.action
                        func_name, args, kwargs = parse_function_call_expr(
                            expr_str, context
                        )
                        return Function(
                            name=func_name,
                            args=args,
                            kwargs=kwargs,
                            thought=expr.thought,
                        )

                dummy = FunctionExperssionToFunction()
                print("FunctionExperssionToFunction")
                # expr.add_successor_map_fn(dummy, map_fn=lambda x: x.data)
                return dummy.forward(expr, context=self.context)

                # expr_str = expr.action
                # func_name, args, kwargs = parse_function_call_expr(expr_str, self.context)
                # return Function(name=func_name, args=args, kwargs=kwargs)
            except Exception as e:
                log.error(f"Error {e} parsing function call expression: {expr}")
                raise ValueError(f"Error {e} parsing function call expression: {expr}")
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
    ) -> Union[FunctionOutput, Function, Parameter]:
        if step == "parse":
            if isinstance(expr_or_fun, Function):
                return expr_or_fun
            return self.parse_func_expr(expr_or_fun)
        else:
            if isinstance(expr_or_fun, Function):
                return self.execute_func(expr_or_fun)
            return self.execute_func_expr(expr_or_fun)

    def forward(
        self,
        expr_or_fun: Union[FunctionExpression, Function, Parameter],
        step: str = "execute",
    ) -> Union[FunctionOutput, Function, Parameter]:
        if isinstance(expr_or_fun, Parameter):
            if step == "execute":
                if isinstance(expr_or_fun.data, Function):
                    return self.execute_func(expr_or_fun)
                else:
                    raise NotImplementedError(
                        "Only function call expressions are supported for now."
                    )
            else:
                if isinstance(expr_or_fun.data, FunctionExpression):
                    return self.parse_func_expr(expr_or_fun)
                else:
                    raise NotImplementedError(
                        f"Only function call expressions are supported for now. Got {expr_or_fun.data}"
                    )
        else:
            return self.call(expr_or_fun=expr_or_fun, step=step)

    def execute_func(
        self, func: Union[Function, Parameter]
    ) -> Union[FunctionOutput, Parameter]:
        r"""Execute the function. If the function is async, use asyncio.run to execute it."""

        if isinstance(func, Parameter):

            class GetFunctionTool(GradComponent):
                def __init__(self):
                    super().__init__()

                def forward(self, func: Parameter, context: Dict[str, object]):
                    return self.bicall(func, context=context)

                def call(self, func: FunctionOutput, context: Dict[str, object]):
                    return self.bicall(func, context=context)

                def bicall(
                    self,
                    func: Union[FunctionOutput, Parameter],
                    context: Dict[str, object] = {},
                ):
                    if isinstance(func, Parameter):
                        printc(f"context: {context}", color="yellow")
                        tool: FunctionTool = context[func.data.name]
                        print(f"tool training: {tool.training}")
                        output = tool.forward(*func.data.args, **func.data.kwargs)
                        # handle the untainable function
                        if not isinstance(output, Parameter):
                            # warnings.info(
                            #     f"Error executing function: {output}", UserWarning
                            # )
                            output = Parameter(
                                name=func.data.name,
                                data=output,
                                eval_input=func.eval_input,
                                requires_opt=False,
                                param_type=ParameterType.OUTPUT,
                            )
                            return output

                        output.predecessors.add(func)
                        return output
                    else:
                        tool: FunctionTool = context[func.name]
                        output = tool.call(*func.args, **func.kwargs)
                        return output

            tool = GetFunctionTool()
            return tool.forward(func, context=self.context)
        else:
            try:
                tool: FunctionTool = self.context[func.name]
                if tool.is_async:
                    return run_async_in_new_loop(tool.acall(*func.args, **func.kwargs))

                else:
                    return tool.call(*func.args, **func.kwargs)
            except Exception as e:
                log.error(f"Error {e} executing function: {func}")
                raise ValueError(f"Error {e} executing function: {func}")

        # try:
        #     tool: FunctionTool = self.context[func.name]
        #     if tool.is_async:
        #         log.debug("Running async function in new loop")
        #         return run_async_in_new_loop(tool.acall(*func.args, **func.kwargs))
        #     else:
        #         # TODO ensure it is set to traing mode
        #         return tool.forward(*func.args, **func.kwargs)
        # except Exception as e:
        #     log.error(f"Error {e} executing function: {func}")
        #     raise ValueError(f"Error {e} executing function: {func}")

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

    def execute_func_expr(
        self, expr: Union[FunctionExpression, Parameter]
    ) -> Union[FunctionOutput, Parameter]:
        r"""Execute the function expression. Support both sync and async functions."""

        if isinstance(expr, Parameter):
            func: Parameter = self.parse_func_expr(expr.data)
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
