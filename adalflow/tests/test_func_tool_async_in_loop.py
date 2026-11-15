"""Regression tests: calling an async FunctionTool synchronously inside a running
event loop must execute the tool, not return an errored FunctionOutput.

The previous implementation, when `is_running_in_event_loop()` was True, scheduled
the coroutine with `loop.create_task(...)` and then passed the resulting `Task`
(not a coroutine) to `asyncio.run_coroutine_threadsafe(...)`. That raised
"A coroutine object is required", which `FunctionTool.call` swallowed into the
`error` field, so the tool silently returned `output=None` instead of running.
"""

import asyncio

from adalflow.core.func_tool import FunctionTool


async def async_double(x):
    await asyncio.sleep(0.001)
    return x * 2


def test_call_async_tool_inside_running_event_loop():
    tool = FunctionTool(fn=async_double)

    async def run():
        return tool.call(21)

    output = asyncio.run(run())

    # The bug returned output=None with error="... A coroutine object is required".
    assert output.error is None, f"unexpected error: {output.error}"
    assert output.output == 42, f"expected 42, got {output.output}"


def test_call_async_tool_outside_event_loop_still_works():
    tool = FunctionTool(fn=async_double)
    output = tool.call(10)
    assert output.error is None, f"unexpected error: {output.error}"
    assert output.output == 20, f"expected 20, got {output.output}"


def test_acall_async_tool_still_works():
    tool = FunctionTool(fn=async_double)
    output = asyncio.run(tool.acall(5))
    assert output.error is None, f"unexpected error: {output.error}"
    assert output.output == 10, f"expected 10, got {output.output}"
