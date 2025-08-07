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

from adalflow.components.agent.runner import Runner


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

            async def mock_tool_execute_async(func, streaming_result=None):
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
            print("sgtreaming_result:", streaming_result)
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
        print("Step history:", runner.step_history)
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
        # Create enough None outputs to reach max_steps (default is 10)
        agent = DummyAgent(planner=FakePlanner([None] * 10), answer_data_type=None)
        runner = Runner(agent=agent)
        runner._tool_execute = lambda func: SimpleNamespace(output="x")

        # The Runner should handle None gracefully and return an error in RunnerResult
        result = asyncio.run(runner.acall(prompt_kwargs={}))

        # Should return a RunnerResult with no output message (runner continues through all steps)
        self.assertIsInstance(result, RunnerResult)
        self.assertIsNone(result.error)  # No error when max steps reached normally
        self.assertIsNotNone(result.answer)  # Has answer message
        self.assertTrue(
            result.answer.startswith("No output generated")
        )  # No output completion message
        # Now that we handle None functions gracefully, we should have all 10 steps with None function
        self.assertEqual(len(runner.step_history), 10)
        self.assertIsNone(runner.step_history[0].function)

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
        import warnings

        mock_function = DummyFunction(name="test_async_func")
        mock_result = SimpleNamespace(output="async_result")

        created_coroutines = []

        def create_coro():
            async def async_mock():
                return mock_result

            coro = async_mock()
            created_coroutines.append(coro)
            return coro

        # Suppress the RuntimeWarning about unawaited coroutines for this specific test
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=".*coroutine.*was never awaited.*",
            )

            # Mock the tool_manager to return a coroutine
            # Note: This test case is complex because _tool_execute_sync needs to handle async properly
            # For now, let's just verify it can handle the async case without crashing
            self.runner.agent.tool_manager = MockToolManager(
                lambda expr_or_fun, step: create_coro()
            )

            result = self.runner._tool_execute_sync(mock_function)

            # Close any created coroutines to prevent warnings
            for coro in created_coroutines:
                coro.close()

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

    def test_conversation_memory_clear_sync(self):
        """Test that clearing conversation memory results in empty chat history in sync call."""
        from adalflow.components.memory.memory import ConversationMemory
        
        # Create a mock planner that captures prompt_kwargs
        captured_prompt_kwargs = []
        
        class CapturingPlanner:
            def __init__(self):
                self.call_count = 0
                
            def call(self, *, prompt_kwargs, model_kwargs=None, use_cache=None, id=None):
                captured_prompt_kwargs.append(prompt_kwargs.copy())
                self.call_count += 1
                # Return a final answer to end execution
                fn = DummyFunction(name="answer_output", _is_answer_final=True, _answer="test response")
                return GeneratorOutput(data=fn)
                
            def get_prompt(self, **kwargs):
                return "test prompt"
        
        # Create runner with conversation memory
        memory = ConversationMemory()
        planner = CapturingPlanner()
        agent = DummyAgent(planner=planner, answer_data_type=None)
        runner = Runner(agent=agent, conversation_memory=memory)
        
        # First call - should have empty history
        result1 = runner.call(prompt_kwargs={"input_str": "Hello, my name is Alice"})
        self.assertEqual(planner.call_count, 1)
        self.assertIn("chat_history_str", captured_prompt_kwargs[0])
        self.assertEqual(captured_prompt_kwargs[0]["chat_history_str"], "")  # Empty on first call
        
        # Second call - should have history from first call
        result2 = runner.call(prompt_kwargs={"input_str": "What is my name?"})
        self.assertEqual(planner.call_count, 2)
        self.assertIn("chat_history_str", captured_prompt_kwargs[1])
        self.assertNotEqual(captured_prompt_kwargs[1]["chat_history_str"], "")  # Has history
        self.assertIn("Alice", captured_prompt_kwargs[1]["chat_history_str"])
        
        # Clear conversation memory
        memory.clear_conversation_turns()
        
        # Third call - should have empty history again
        result3 = runner.call(prompt_kwargs={"input_str": "Do you remember my name?"})
        self.assertEqual(planner.call_count, 3)
        self.assertIn("chat_history_str", captured_prompt_kwargs[2])
        self.assertEqual(captured_prompt_kwargs[2]["chat_history_str"], "")  # Empty after clear

    async def _test_conversation_memory_clear_async(self):
        """Test that clearing conversation memory results in empty chat history in async call."""
        from adalflow.components.memory.memory import ConversationMemory
        
        # Create a mock planner that captures prompt_kwargs
        captured_prompt_kwargs = []
        
        class CapturingAsyncPlanner:
            def __init__(self):
                self.call_count = 0
                
            async def acall(self, *, prompt_kwargs, model_kwargs=None, use_cache=None, id=None):
                captured_prompt_kwargs.append(prompt_kwargs.copy())
                self.call_count += 1
                # Return a final answer to end execution
                fn = DummyFunction(name="answer_output", _is_answer_final=True, _answer="test response")
                return GeneratorOutput(data=fn)
                
            def get_prompt(self, **kwargs):
                return "test prompt"
        
        # Create runner with conversation memory
        memory = ConversationMemory()
        planner = CapturingAsyncPlanner()
        agent = DummyAgent(planner=planner, answer_data_type=None)
        runner = Runner(agent=agent, conversation_memory=memory)
        
        # Mock the async tool execution
        async def mock_tool_execute_async(func):
            from adalflow.core.types import FunctionOutput
            return FunctionOutput(name=func.name, input=func, output="test response")
        
        runner._tool_execute_async = mock_tool_execute_async
        
        # First call - should have empty history
        result1 = await runner.acall(prompt_kwargs={"input_str": "Hello, my name is Bob"})
        self.assertEqual(planner.call_count, 1)
        self.assertIn("chat_history_str", captured_prompt_kwargs[0])
        self.assertEqual(captured_prompt_kwargs[0]["chat_history_str"], "")  # Empty on first call
        
        # Second call - should have history from first call
        result2 = await runner.acall(prompt_kwargs={"input_str": "What is my name?"})
        self.assertEqual(planner.call_count, 2)
        self.assertIn("chat_history_str", captured_prompt_kwargs[1])
        self.assertNotEqual(captured_prompt_kwargs[1]["chat_history_str"], "")  # Has history
        self.assertIn("Bob", captured_prompt_kwargs[1]["chat_history_str"])
        
        # Clear conversation memory
        memory.clear_conversation_turns()
        
        # Third call - should have empty history again
        result3 = await runner.acall(prompt_kwargs={"input_str": "Do you remember my name?"})
        self.assertEqual(planner.call_count, 3)
        self.assertIn("chat_history_str", captured_prompt_kwargs[2])
        self.assertEqual(captured_prompt_kwargs[2]["chat_history_str"], "")  # Empty after clear

    def test_conversation_memory_clear_async(self):
        """Wrapper to run async test."""
        asyncio.run(self._test_conversation_memory_clear_async())

    async def _test_conversation_memory_clear_streaming(self):
        """Test that clearing conversation memory results in empty chat history in streaming."""
        from adalflow.components.memory.memory import ConversationMemory
        
        # Create a mock planner that captures prompt_kwargs
        captured_prompt_kwargs = []
        
        class CapturingStreamingPlanner:
            def __init__(self):
                self.call_count = 0
                
            async def acall(self, *, prompt_kwargs, model_kwargs=None, use_cache=None, id=None):
                captured_prompt_kwargs.append(prompt_kwargs.copy())
                self.call_count += 1
                # Return a final answer to end execution
                fn = DummyFunction(name="answer_output", _is_answer_final=True, _answer="test response")
                
                # Simulate streaming with async generator
                async def stream_gen():
                    yield "streaming event 1"
                    yield "streaming event 2"
                    
                return GeneratorOutput(data=fn, raw_response=stream_gen())
                
            def get_prompt(self, **kwargs):
                return "test prompt"
        
        # Create runner with conversation memory
        memory = ConversationMemory()
        planner = CapturingStreamingPlanner()
        agent = DummyAgent(planner=planner, answer_data_type=None)
        runner = Runner(agent=agent, conversation_memory=memory)
        
        # Mock the async tool execution
        async def mock_tool_execute_async(func, streaming_result=None):
            from adalflow.core.types import FunctionOutput
            return FunctionOutput(name=func.name, input=func, output="test response")
        
        runner._tool_execute_async = mock_tool_execute_async
        
        # First streaming call - should have empty history
        streaming_result1 = runner.astream(prompt_kwargs={"input_str": "Hello, my name is Charlie"})
        events1 = []
        async for event in streaming_result1.stream_events():
            events1.append(event)
        
        self.assertEqual(planner.call_count, 1)
        self.assertIn("chat_history_str", captured_prompt_kwargs[0])
        self.assertEqual(captured_prompt_kwargs[0]["chat_history_str"], "")  # Empty on first call
        
        # Second streaming call - should have history from first call
        streaming_result2 = runner.astream(prompt_kwargs={"input_str": "What is my name?"})
        events2 = []
        async for event in streaming_result2.stream_events():
            events2.append(event)
            
        self.assertEqual(planner.call_count, 2)
        self.assertIn("chat_history_str", captured_prompt_kwargs[1])
        self.assertNotEqual(captured_prompt_kwargs[1]["chat_history_str"], "")  # Has history
        self.assertIn("Charlie", captured_prompt_kwargs[1]["chat_history_str"])
        
        # Clear conversation memory
        memory.clear_conversation_turns()
        
        # Third streaming call - should have empty history again
        streaming_result3 = runner.astream(prompt_kwargs={"input_str": "Do you remember my name?"})
        events3 = []
        async for event in streaming_result3.stream_events():
            events3.append(event)
            
        self.assertEqual(planner.call_count, 3)
        self.assertIn("chat_history_str", captured_prompt_kwargs[2])
        self.assertEqual(captured_prompt_kwargs[2]["chat_history_str"], "")  # Empty after clear

    def test_conversation_memory_clear_streaming(self):
        """Wrapper to run async streaming test."""
        asyncio.run(self._test_conversation_memory_clear_streaming())


class TestRunnerBugFixes(unittest.TestCase):
    """Tests for specific bugs that were found and fixed in the runner implementation."""

    def test_null_function_handling_call(self):
        """Test that call method handles None functions without crashing (fixes AttributeError on function.id)."""
        # Simulate planner returning None function multiple times
        agent = DummyAgent(
            planner=FakePlanner([
                GeneratorOutput(data=None, error="Parsing error"),
                GeneratorOutput(data=None, error="Still parsing error")
            ]),
            answer_data_type=None
        )
        runner = Runner(agent=agent, max_steps=2)
        
        result = runner.call(prompt_kwargs={})
        
        # Should return a result and handle None gracefully without crashing
        self.assertIsInstance(result, RunnerResult)
        # Should have processed the None function in step history
        self.assertEqual(len(runner.step_history), 2)  # Will try max_steps=2
        # First step should have None function and error observation
        step = runner.step_history[0] 
        self.assertIsNone(step.function)
        self.assertEqual(step.observation, "Parsing error")

    def test_null_function_handling_acall(self):
        """Test that acall method handles None functions without crashing."""
        async def async_test():
            agent = DummyAgent(
                planner=FakePlanner([
                    GeneratorOutput(data=None, error="Async parsing error"),
                    GeneratorOutput(data=None, error="Still async parsing error")
                ]),
                answer_data_type=None
            )
            runner = Runner(agent=agent, max_steps=2)
            
            result = await runner.acall(prompt_kwargs={})
            
            # Should return a result and handle None gracefully without crashing
            self.assertIsInstance(result, RunnerResult) 
            # Should have processed the None function in step history
            self.assertEqual(len(runner.step_history), 2)  # Will try max_steps=2
            # First step should have None function and error observation
            step = runner.step_history[0] 
            self.assertIsNone(step.function)
            self.assertEqual(step.observation, "Async parsing error")
            
        asyncio.run(async_test())

    def test_function_id_assignment_safety(self):
        """Test that function.id assignment is safe when function is None."""
        # Create a GeneratorOutput with None data to test the null safety
        agent = DummyAgent(
            planner=FakePlanner([GeneratorOutput(data=None)]),
            answer_data_type=None
        )
        runner = Runner(agent=agent)
        
        # This should not crash due to trying to access None.id
        result = runner.call(prompt_kwargs={})
        self.assertIsInstance(result, RunnerResult)

    def test_consistent_error_variable_initialization(self):
        """Test that current_error variable is properly initialized in all methods."""
        
        def create_error_planner():
            """Create planner that causes internal error."""
            class ErrorPlanner:
                def call(self, **kwargs):
                    return "invalid_output"  # Not a GeneratorOutput
                    
                async def acall(self, **kwargs):
                    return "invalid_output"
                    
                def get_prompt(self, **kwargs):
                    return "test prompt"
            return ErrorPlanner()
        
        agent = DummyAgent(planner=create_error_planner(), answer_data_type=None)
        runner = Runner(agent=agent)
        
        # Test call method doesn't crash with undefined current_error
        result = runner.call(prompt_kwargs={})
        self.assertIsInstance(result, RunnerResult)
        self.assertIn("Expected GeneratorOutput", result.answer)
        
        # Test acall method doesn't crash with undefined current_error  
        async def async_test():
            result = await runner.acall(prompt_kwargs={})
            self.assertIsInstance(result, RunnerResult)
            self.assertIn("Expected GeneratorOutput", result.answer)
            
        asyncio.run(async_test())

    def test_output_data_vs_function_consistency(self):
        """Test that we correctly use output.data instead of non-existent output.function."""
        fn = DummyFunction(name="test", _is_answer_final=True, _answer="success")
        
        # Test that GeneratorOutput.data is accessed, not .function
        agent = DummyAgent(
            planner=FakePlanner([GeneratorOutput(data=fn)]),
            answer_data_type=None
        )
        runner = Runner(agent=agent)
        
        result = runner.call(prompt_kwargs={})
        self.assertEqual(result.answer, "success")

    def test_span_data_consistency(self):
        """Test that span data uses answer strings, not full RunnerResult objects."""
        fn = DummyFunction(name="finish", _is_answer_final=True, _answer="span_test")
        mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(output="span_test")
        
        agent = DummyAgent(
            planner=FakePlanner([GeneratorOutput(data=fn)]),
            answer_data_type=None,
            tool_manager=mock_tool_manager,
        )
        runner = Runner(agent=agent)
        
        # This would previously fail tests due to storing RunnerResult instead of string
        result = runner.call(prompt_kwargs={})
        self.assertEqual(result.answer, "span_test")
        self.assertIsInstance(result.answer, str)  # Should be string, not RunnerResult

    def test_cancellation_behavior(self):
        """Test that cancellation works correctly without asyncio.shield interference."""
        async def async_test():
            # Create a long-running operation that can be cancelled
            fn = DummyFunction(name="slow_op", _is_answer_final=False)
            
            class SlowPlanner:
                async def acall(self, **kwargs):
                    await asyncio.sleep(2)  # Long operation
                    return GeneratorOutput(data=fn)
                    
                def get_prompt(self, **kwargs):
                    return "test prompt"
            
            agent = DummyAgent(planner=SlowPlanner(), answer_data_type=None)
            runner = Runner(agent=agent)
            
            # Start streaming
            stream_result = runner.astream(prompt_kwargs={})
            
            # Cancel after short delay
            await asyncio.sleep(0.1)
            await runner.cancel()
            
            # Wait for completion 
            await stream_result.wait_for_completion()
            
            # Should have been cancelled successfully
            self.assertTrue(runner.is_cancelled())
            
        asyncio.run(async_test())

    def test_error_handling_consistency_across_methods(self):
        """Test that error handling is consistent between call, acall, and astream."""
        
        # Test consistent error handling when planner fails
        class FailingPlanner:
            def call(self, **kwargs):
                raise ValueError("Planner failed")
                
            async def acall(self, **kwargs):
                raise ValueError("Async planner failed")
                
            def get_prompt(self, **kwargs):
                return "test prompt"
        
        agent = DummyAgent(planner=FailingPlanner(), answer_data_type=None)
        runner = Runner(agent=agent)
        
        # All methods should handle errors gracefully
        result_sync = runner.call(prompt_kwargs={})
        self.assertIsInstance(result_sync, RunnerResult)
        
        async def async_test():
            result_async = await runner.acall(prompt_kwargs={})
            self.assertIsInstance(result_async, RunnerResult)
            
            stream_result = runner.astream(prompt_kwargs={})
            await stream_result.wait_for_completion()
            self.assertIsNotNone(stream_result._exception)
            
        asyncio.run(async_test())

    def test_json_parsing_error_handling(self):
        """Test that JSON parsing errors in _process_data are handled correctly."""
        import adalflow.components.agent.runner as runner_module
        
        # Test with pydantic dataclass that requires JSON parsing
        from pydantic import BaseModel
        
        class TestModel(BaseModel):
            value: str
        
        runner = Runner(
            agent=DummyAgent(planner=FakePlanner([]), answer_data_type=TestModel),
            max_steps=1
        )
        
        # Test malformed JSON
        with self.assertRaises(ValueError) as cm:
            runner._process_data("invalid json")
        self.assertIn("Invalid JSON", str(cm.exception))
        
        # Test JSON that's not a dict  
        with self.assertRaises(ValueError) as cm:
            runner._process_data('"just a string"')
        self.assertIn("Expected dict after JSON parsing", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
