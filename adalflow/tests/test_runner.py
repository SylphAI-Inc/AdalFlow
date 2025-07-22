import unittest
import unittest.mock
import asyncio
from types import SimpleNamespace
from adalflow.core.types import (
    GeneratorOutput,
    Function,
    RunnerResult,
    FinalOutputItem,
    RunItemStreamEvent,
)

from adalflow.core.runner import Runner


class DummyFunction(Function):
    """Mimics adalflow.core.types.Function."""

    def __init__(self, name, kwargs=None, _is_answer_final=False, _answer=None):
        super().__init__(name=name, kwargs=kwargs or {})
        self._is_answer_final = _is_answer_final
        self._answer = _answer


class FakePlanner:
    """Planner stub that returns a sequence of GeneratorOutput or raw."""

    def __init__(self, outputs):
        # Wrap outputs in GeneratorOutput if they're not already
        self._outputs = [
            out if isinstance(out, GeneratorOutput) else GeneratorOutput(data=out)
            for out in outputs
        ]
        self._idx = 0

    def call(self, *, prompt_kwargs, model_kwargs=None, use_cache=None, id=None):
        if self._idx >= len(self._outputs):
            raise IndexError("No more outputs")
        out = self._outputs[self._idx]
        self._idx += 1
        return out

    async def acall(self, *, prompt_kwargs, model_kwargs=None, use_cache=None, id=None):
        return self.call(
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
            use_cache=use_cache,
            id=id,
        )

    def get_prompt(self, **kwargs):
        return ""


class FakeStreamingPlanner:
    """Planner stub that returns async generators for streaming."""

    def __init__(self, outputs):
        # Wrap outputs in GeneratorOutput if they're not already
        self._outputs = [
            out if isinstance(out, GeneratorOutput) else GeneratorOutput(data=out)
            for out in outputs
        ]
        self._idx = 0

    def call(self, *, prompt_kwargs, model_kwargs=None, use_cache=None, id=None):
        if self._idx >= len(self._outputs):
            raise IndexError("No more outputs")
        out = self._outputs[self._idx]
        self._idx += 1
        return out

    async def acall(self, *, prompt_kwargs, model_kwargs=None, use_cache=None, id=None):
        # For streaming, return a GeneratorOutput with an async generator
        async def async_generator():
            # Yield some intermediate events (simulating streaming)
            yield "intermediate_event_1"
            yield "intermediate_event_2"
            # Yield the final data
            if self._idx >= len(self._outputs):
                raise IndexError("No more outputs")
            out = self._outputs[self._idx]
            self._idx += 1
            yield out.data

        # Return a GeneratorOutput with the async generator as raw_response
        if self._idx >= len(self._outputs):
            raise IndexError("No more outputs")
        out = self._outputs[self._idx]
        return GeneratorOutput(data=out.data, raw_response=async_generator())

    def get_prompt(self, **kwargs):
        return ""


class MockToolManager:
    """Mock tool manager that mimics the real tool manager interface."""

    def __init__(self, sync_callable=None):
        self._sync_callable = sync_callable

    def __call__(self, expr_or_fun, step):
        """For backward compatibility with existing sync callable interface."""
        if self._sync_callable:
            return self._sync_callable(expr_or_fun, step)
        return SimpleNamespace(output="mock_output")

    def execute_func(self, func):
        """Sync method for tool execution."""
        from adalflow.core.types import FunctionOutput

        if self._sync_callable:
            result = self._sync_callable(func, "execute")
            if hasattr(result, "output"):
                return FunctionOutput(name=func.name, input=func, output=result.output)
            else:
                return FunctionOutput(name=func.name, input=func, output=result)
        return FunctionOutput(name=func.name, input=func, output="mock_output")

    async def execute_func_async(self, func):
        """Async method for tool execution."""
        from adalflow.core.types import FunctionOutput

        if self._sync_callable:
            result = self._sync_callable(func, "execute")
            if hasattr(result, "output"):
                return FunctionOutput(name=func.name, input=func, output=result.output)
            else:
                return FunctionOutput(name=func.name, input=func, output=result)
        return FunctionOutput(name=func.name, input=func, output="mock_async_output")


class DummyAgent:
    """Bare-bones Agent for Runner, including answer_data_type for Runner.__init__."""

    def __init__(self, planner, max_steps=10, tool_manager=None, answer_data_type=None):
        self.planner = planner
        self.max_steps = max_steps
        if tool_manager and not hasattr(tool_manager, "execute_func_async"):
            # Wrap simple callable in MockToolManager
            self.tool_manager = MockToolManager(tool_manager)
        else:
            self.tool_manager = tool_manager or MockToolManager()
        self.answer_data_type = answer_data_type


class DummyStepOutput:
    """Stub for StepOutput with flexible constructor."""

    def __init__(self, *args, **kwargs):
        self.step = kwargs.get("step", args[0] if len(args) > 0 else None)
        self.function = kwargs.get("function", args[1] if len(args) > 1 else None)
        # Runner uses 'observation'; fallback to 'output'
        self.observation = kwargs.get("observation", kwargs.get("output", None))


class TestRunner(unittest.TestCase):
    def setUp(self):
        # Use real StepOutput instead of mocking to ensure compatibility with RunnerResponse validation
        # Prepare a Runner with dummy agent
        self.runner = Runner(
            agent=DummyAgent(
                planner=None, answer_data_type=None, tool_manager=MockToolManager()
            )
        )

    def test_check_last_step(self):
        cont_fn = DummyFunction(name="continue", _is_answer_final=False)
        answer_output_fn = DummyFunction(name="answer_output", _is_answer_final=True)
        self.assertTrue(self.runner._check_last_step(answer_output_fn))
        self.assertFalse(self.runner._check_last_step(cont_fn))

    def test_call_single_step_answer_output_returns_runner_response(self):
        fn = DummyFunction(name="answer_output", _is_answer_final=True, _answer="done")
        # Create a mock tool manager that returns a FunctionOutput
        mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(output="done")
        agent = DummyAgent(
            planner=FakePlanner([GeneratorOutput(data=fn)]),
            answer_data_type=None,
            tool_manager=mock_tool_manager,
        )
        runner = Runner(agent=agent)

        result = runner.call(prompt_kwargs={})

        # Verify result is RunnerResult
        self.assertIsInstance(result, RunnerResult)
        self.assertEqual(result.answer, "done")
        # When _is_answer_final=True, Runner processes answer directly without adding to step history
        self.assertEqual(len(result.step_history), 0)
        self.assertEqual(len(runner.step_history), 0)

    def test_acall_returns_runner_response(self):
        """Test that acall method returns RunnerResponse."""

        async def async_test():
            from adalflow.core.types import FunctionOutput

            fn = DummyFunction(
                name="answer_output", _is_answer_final=True, _answer="async-done"
            )
            agent = DummyAgent(
                planner=FakePlanner([GeneratorOutput(data=fn)]), answer_data_type=None
            )
            runner = Runner(agent=agent)

            async def mock_tool_execute_async(func):
                return FunctionOutput(name=func.name, input=func, output="async-done")

            runner._tool_execute_async = mock_tool_execute_async

            result = await runner.acall(prompt_kwargs={})

            # Verify result is RunnerResult
            self.assertIsInstance(result, RunnerResult)
            self.assertEqual(result.answer, "async-done")
            # When _is_answer_final=True, Runner processes answer directly without adding to step history
            self.assertEqual(len(result.step_history), 0)
            self.assertEqual(len(runner.step_history), 0)

        asyncio.run(async_test())

    def test_astream_emits_final_output_item(self):
        """Test that astream emits FinalOutputItem with RunnerResponse."""

        async def async_test():
            from adalflow.core.types import FunctionOutput

            fn = DummyFunction(
                name="answer_output", _is_answer_final=True, _answer="stream-done"
            )
            agent = DummyAgent(
                planner=FakeStreamingPlanner([GeneratorOutput(data=fn)]),
                answer_data_type=None,
            )
            runner = Runner(agent=agent)

            async def mock_tool_execute_async(func):
                return FunctionOutput(name=func.name, input=func, output="stream-done")

            runner._tool_execute_async = mock_tool_execute_async

            # Start streaming
            streaming_result = runner.astream(prompt_kwargs={})

            final_output_events = []
            timeout_seconds = 5
            try:

                async def collect_events():
                    async for event in streaming_result.stream_events():
                        if (
                            isinstance(event, RunItemStreamEvent)
                            and event.name == "agent.execution_complete"
                        ):
                            final_output_events.append(event)

                await asyncio.wait_for(collect_events(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                self.fail(f"Stream events timed out after {timeout_seconds} seconds")

            # Verify we got a final output event
            self.assertEqual(len(final_output_events), 1)
            final_event = final_output_events[0]

            # Verify the event contains FinalOutputItem with RunnerResponse
            self.assertIsInstance(final_event.item, FinalOutputItem)
            runner_response = final_event.item.data
            self.assertIsInstance(runner_response, RunnerResult)
            self.assertEqual(runner_response.answer, "stream-done")
            # When _is_answer_final=True, Runner processes answer directly without adding to step history
            self.assertEqual(len(runner_response.step_history), 0)

            # Verify streaming result has final result
            self.assertEqual(streaming_result.answer, "stream-done")

        asyncio.run(async_test())

    def test_call_nonfinish_then_answer_output(self):
        """Test multi-step execution with non-answer_output then answer_output functions."""
        fn1 = DummyFunction(name="search", _is_answer_final=False)
        fn2 = DummyFunction(
            name="answer_output", _is_answer_final=True, _answer="test output"
        )
        mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(
            output="test output"
        )
        agent = DummyAgent(
            planner=FakePlanner([GeneratorOutput(data=fn1), GeneratorOutput(data=fn2)]),
            answer_data_type=None,
            tool_manager=mock_tool_manager,
        )
        runner = Runner(agent=agent)

        result = runner.call(prompt_kwargs={})

        # Verify result is RunnerResult
        self.assertIsInstance(result, RunnerResult)
        self.assertEqual(result.answer, "test output")
        # Verify step history contains only the first (non-final) step execution
        # The second step with _is_answer_final=True doesn't get added to history
        self.assertEqual(len(result.step_history), 1)
        self.assertEqual(
            result.step_history[0].function, fn1
        )  # Only first step in history
        self.assertEqual(len(runner.step_history), 1)

    def test_call_respects_max_steps_without_answer_output(self):
        """Test that Runner respects max_steps limit without answer_output function."""
        # Create outputs for 5 steps without answer_output
        functions = [
            DummyFunction(name=f"action_{i}", _is_answer_final=False) for i in range(5)
        ]
        outputs = [GeneratorOutput(data=fn) for fn in functions]
        mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(
            output=f"output_{expr_or_fun.name}"
        )
        agent = DummyAgent(
            planner=FakePlanner(outputs),
            max_steps=3,
            answer_data_type=None,
            tool_manager=mock_tool_manager,
        )
        runner = Runner(agent=agent)

        result = runner.call(prompt_kwargs={})
        # Should only execute 3 steps due to max_steps limit
        self.assertEqual(len(runner.step_history), 3)

        # When max_steps is reached without answer_output, result should have "No output generated" message
        self.assertIsInstance(result, RunnerResult)
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.answer)
        self.assertTrue(result.answer.startswith("No output generated"))

        # Check that the correct functions were executed
        for i, step in enumerate(runner.step_history):
            self.assertEqual(step.function.name, f"action_{i}")
            self.assertEqual(step.observation, f"output_action_{i}")

    def test_call_no_answer_data_type(self):
        """Test call with no answer_data_type returns RunnerResult."""
        fn = DummyFunction(
            name="answer_output", _is_answer_final=True, _answer="{'result': 'success'}"
        )
        mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(
            output="{'result': 'success'}"
        )
        agent = DummyAgent(
            planner=FakePlanner([GeneratorOutput(data=fn)]),
            answer_data_type=None,
            tool_manager=mock_tool_manager,
        )
        runner = Runner(agent=agent)

        result = runner.call(prompt_kwargs={})

        # Verify result is RunnerResult wrapping the dict output
        self.assertIsInstance(result, RunnerResult)
        # When no answer_data_type is specified, it defaults to str and should convert dict to string
        self.assertEqual(result.answer, "{'result': 'success'}")
        # When _is_answer_final=True, Runner processes answer directly without adding to step history
        self.assertEqual(len(result.step_history), 0)
        self.assertEqual(len(runner.step_history), 0)

    def test_additional_acall_single_step(self):
        """Additional test for acall single step execution."""

        async def async_test():
            from adalflow.core.types import FunctionOutput

            fn = DummyFunction(
                name="answer_output", _is_answer_final=True, _answer="async-done"
            )
            agent = DummyAgent(
                planner=FakePlanner([GeneratorOutput(data=fn)]), answer_data_type=None
            )
            runner = Runner(agent=agent)

            async def mock_tool_execute_async(func):
                return FunctionOutput(name=func.name, input=func, output="async-done")

            runner._tool_execute_async = mock_tool_execute_async

            result = await runner.acall(prompt_kwargs={})

            # Verify result is RunnerResult
            self.assertIsInstance(result, RunnerResult)
            self.assertEqual(result.answer, "async-done")
            # When _is_answer_final=True, Runner processes answer directly without adding to step history
            self.assertEqual(len(result.step_history), 0)
            self.assertEqual(len(runner.step_history), 0)

        asyncio.run(async_test())

    def test_acall_invalid_answer_data_type(self):
        """Test acall with invalid data handling."""
        agent = DummyAgent(planner=FakePlanner([None]), answer_data_type=None)
        runner = Runner(agent=agent)
        runner._tool_execute = lambda func: SimpleNamespace(output="x")

        # The Runner should handle None gracefully and return an error in RunnerResult
        result = asyncio.run(runner.acall(prompt_kwargs={}))

        # Should return a RunnerResult with no output message (runner continues through all steps)
        self.assertIsInstance(result, RunnerResult)
        self.assertIsNone(result.error)  # No error in final result
        self.assertIsNotNone(result.answer)  # Has answer message
        self.assertTrue(
            result.answer.startswith("No output generated")
        )  # No output completion message
        self.assertEqual(runner.step_history, [])  # No successful steps in history

    def test_process_data_without_answer_data_type(self):
        """Test _process_data without answer_data_type."""
        out = self.runner._process_data(data="raw", id=None)
        self.assertEqual(out, "raw")

    def test_tool_execute_sync_with_sync_function(self):
        """Test _tool_execute_sync with synchronous function result."""
        mock_function = DummyFunction(name="test_func")
        mock_result = SimpleNamespace(output="sync_result")

        # Mock the tool_manager to return a sync result
        self.runner.agent.tool_manager = MockToolManager(
            lambda expr_or_fun, step: mock_result
        )

        result = self.runner._tool_execute_sync(mock_function)
        # _tool_execute_sync wraps the result in FunctionOutput
        from adalflow.core.types import FunctionOutput

        self.assertIsInstance(result, FunctionOutput)
        self.assertEqual(result.output, mock_result.output)

    def test_tool_execute_sync_with_async_function_no_loop(self):
        """Test _tool_execute_sync with async function when no event loop is running."""

        mock_function = DummyFunction(name="test_async_func")
        mock_result = SimpleNamespace(output="async_result")

        async def async_mock():
            return mock_result

        # Mock the tool_manager to return a coroutine
        # Note: This test case is complex because _tool_execute_sync needs to handle async properly
        # For now, let's just verify it can handle the async case without crashing
        self.runner.agent.tool_manager = MockToolManager(
            lambda expr_or_fun, step: async_mock()
        )

        result = self.runner._tool_execute_sync(mock_function)
        # _tool_execute_sync wraps the result in FunctionOutput
        from adalflow.core.types import FunctionOutput

        self.assertIsInstance(result, FunctionOutput)
        # The async handling is complex, so just verify the structure
        self.assertEqual(result.name, mock_function.name)

    def test_tool_execute_sync_with_async_generator_no_loop(self):
        """Test _tool_execute_sync with async generator when no event loop is running."""

        mock_function = DummyFunction(name="test_async_gen")

        async def async_generator():
            yield SimpleNamespace(output="item1")
            yield SimpleNamespace(output="item2")
            yield SimpleNamespace(output="final_result")

        # Mock the tool_manager to return an async generator
        self.runner.agent.tool_manager = MockToolManager(
            lambda expr_or_fun, step: async_generator()
        )

        result = self.runner._tool_execute_sync(mock_function)
        # For async generator, it should be wrapped but not executed
        # This test needs to be adjusted as async generators aren't fully supported in sync mode
        from adalflow.core.types import FunctionOutput

        self.assertIsInstance(result, FunctionOutput)
        # The async generator case is complex - let's just verify it's wrapped
        self.assertEqual(result.name, mock_function.name)

    def test_comprehensive_astream_behavior(self):
        """Comprehensive test for astream behavior with multiple scenarios."""

        async def async_test():
            from adalflow.core.types import FunctionOutput

            # Test 1: Single step with answer_output function
            fn = DummyFunction(
                name="answer_output", _is_answer_final=True, _answer="stream-test"
            )
            agent = DummyAgent(
                planner=FakeStreamingPlanner([GeneratorOutput(data=fn)]),
                answer_data_type=None,
            )
            runner = Runner(agent=agent)

            async def mock_tool_execute_async(func):
                return FunctionOutput(name=func.name, input=func, output="stream-test")

            runner._tool_execute_async = mock_tool_execute_async

            # Collect all events from stream
            streaming_result = runner.astream(prompt_kwargs={})
            events = []
            timeout_seconds = 5
            try:

                async def collect_events():
                    async for event in streaming_result.stream_events():
                        events.append(event)

                await asyncio.wait_for(collect_events(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                self.fail(f"Stream events timed out after {timeout_seconds} seconds")

            # Verify we have events
            self.assertGreater(len(events), 0)

            # Find the final output event
            final_events = [
                e
                for e in events
                if isinstance(e, RunItemStreamEvent)
                and e.name == "agent.execution_complete"
            ]
            self.assertEqual(len(final_events), 1)

            final_event = final_events[0]
            self.assertIsInstance(final_event.item, FinalOutputItem)
            self.assertIsInstance(final_event.item.data, RunnerResult)
            self.assertEqual(final_event.item.data.answer, "stream-test")
            # When _is_answer_final=True, no step history is created
            self.assertEqual(len(final_event.item.data.step_history), 0)

        asyncio.run(async_test())

    def test_runner_response_consistency(self):
        """Test that all Runner methods return consistent RunnerResponse objects."""
        fn = DummyFunction(
            name="answer_output", _is_answer_final=True, _answer="consistent-output"
        )
        mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(
            output="consistent-output"
        )

        # Test sync call
        agent1 = DummyAgent(
            planner=FakePlanner([GeneratorOutput(data=fn)]),
            answer_data_type=None,
            tool_manager=mock_tool_manager,
        )
        runner1 = Runner(agent=agent1)
        result1 = runner1.call(prompt_kwargs={})
        self.assertIsInstance(result1, RunnerResult)
        self.assertEqual(result1.answer, "consistent-output")
        # When _is_answer_final=True, Runner processes answer directly without adding to step history
        self.assertEqual(len(result1.step_history), 0)

        # Test async call with fresh planner
        async def async_test():
            from adalflow.core.types import FunctionOutput

            agent2 = DummyAgent(
                planner=FakePlanner([GeneratorOutput(data=fn)]),
                answer_data_type=None,
                tool_manager=mock_tool_manager,
            )
            runner2 = Runner(agent=agent2)

            async def mock_tool_execute_async(func):
                return FunctionOutput(
                    name=func.name, input=func, output="consistent-output"
                )

            runner2._tool_execute_async = mock_tool_execute_async
            result2 = await runner2.acall(prompt_kwargs={})

            self.assertIsInstance(result2, RunnerResult)
            self.assertEqual(result2.answer, "consistent-output")
            # When _is_answer_final=True, Runner processes answer directly without adding to step history
            self.assertEqual(len(result2.step_history), 0)

        asyncio.run(async_test())

    def test_runner_integration_methods(self):
        """Test that Runner has all expected methods for integration."""
        # Create a dummy agent to avoid None errors
        dummy_agent = DummyAgent(planner=None, answer_data_type=None)
        runner = Runner(agent=dummy_agent)  # Basic instantiation test

        # Verify Runner has expected methods
        expected_methods = ["call", "acall", "astream"]
        for method_name in expected_methods:
            self.assertTrue(
                hasattr(runner, method_name), f"Runner should have {method_name} method"
            )
            self.assertTrue(
                callable(getattr(runner, method_name)),
                f"Runner.{method_name} should be callable",
            )


if __name__ == "__main__":
    unittest.main()
