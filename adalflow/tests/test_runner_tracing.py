import unittest
import os
import asyncio
from unittest.mock import patch
from types import SimpleNamespace
from typing import Any, List

from adalflow.components.agent.runner import Runner
from adalflow.core.types import (
    GeneratorOutput,
    Function,
    RunnerResult,
    FunctionOutput,
)
from adalflow.tracing import (
    trace,
    set_tracing_disabled,
    add_trace_processor,
    set_trace_processors,
    GLOBAL_TRACE_PROVIDER,
    generator_span,
)
from adalflow.tracing.span_data import (
    AdalFlowRunnerSpanData,
    AdalFlowGeneratorSpanData,
    AdalFlowToolSpanData,
    AdalFlowResponseSpanData,
    AdalFlowStepSpanData,
)
from adalflow.tracing.processor_interface import TracingProcessor


class MockTracingProcessor(TracingProcessor):
    """Mock tracing processor for testing."""

    def __init__(self):
        self.trace_starts = []
        self.trace_ends = []
        self.span_starts = []
        self.span_ends = []
        self.shutdown_called = False
        self.flush_called = False

    def on_trace_start(self, trace) -> None:
        self.trace_starts.append(trace)

    def on_trace_end(self, trace) -> None:
        self.trace_ends.append(trace)

    def on_span_start(self, span) -> None:
        self.span_starts.append(span)

    def on_span_end(self, span) -> None:
        self.span_ends.append(span)

    def shutdown(self) -> None:
        self.shutdown_called = True

    def force_flush(self) -> None:
        self.flush_called = True

    def get_spans_by_type(self, span_type: str) -> List[Any]:
        """Helper to get spans by their data type."""
        return [span for span in self.span_starts if span.span_data.type == span_type]

    def get_spans_by_name(self, name: str) -> List[Any]:
        """Helper to get spans by their name."""
        return [span for span in self.span_starts if span.span_data.name == name]


class DummyFunction(Function):
    """Mimics adalflow.core.types.Function."""

    def __init__(self, name, kwargs=None, _is_answer_final=False, _answer=None):
        super().__init__(name=name, kwargs=kwargs or {})
        self._is_answer_final = _is_answer_final
        self._answer = _answer


class FakePlanner:
    """Planner stub that returns a sequence of GeneratorOutput."""

    def __init__(self, outputs):
        self._outputs = [
            out if isinstance(out, GeneratorOutput) else GeneratorOutput(data=out)
            for out in outputs
        ]
        self._idx = 0

    def call(self, *, prompt_kwargs, model_kwargs=None, use_cache=None, id=None):
        with generator_span(
            generator_id="fake_planner" + (id if id else ""),
            model_kwargs=model_kwargs or {},
            prompt_kwargs=prompt_kwargs or {},
            prompt_template_with_keywords="fake template",
        ) as generator_span_data:
            if self._idx >= len(self._outputs):
                # Return a finish function if we run out of outputs
                output = GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="default_finish"
                    )
                )
            else:
                output = self._outputs[self._idx]
                self._idx += 1

            # Update span with fake response data
            generator_span_data.span_data.update_attributes(
                {
                    "raw_response": str(output.data),
                    "final_response": output,
                }
            )
            return output

    async def acall(self, *, prompt_kwargs, model_kwargs=None, use_cache=None, id=None):
        with generator_span(
            generator_id="async_planner" + (id if id else ""),
            model_kwargs=model_kwargs or {},
            prompt_kwargs=prompt_kwargs or {},
            prompt_template_with_keywords="fake async template",
        ) as generator_span_data:
            if self._idx >= len(self._outputs):
                # Return a finish function if we run out of outputs
                output = GeneratorOutput(
                    data=DummyFunction(
                        name="finish", _is_answer_final=True, _answer="default_finish"
                    )
                )
            else:
                output = self._outputs[self._idx]
                self._idx += 1

            # Update span with fake response data
            generator_span_data.span_data.update_attributes(
                {
                    "raw_response": str(output.data),
                    "final_response": output,
                }
            )
            return output

    def get_prompt(self, **kwargs):
        return "test prompt"


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
        if self._sync_callable:
            result = self._sync_callable(func, "execute")
            if hasattr(result, "output"):
                return FunctionOutput(name=func.name, input=func, output=result.output)
            else:
                return FunctionOutput(name=func.name, input=func, output=result)
        return FunctionOutput(name=func.name, input=func, output="mock_output")

    async def execute_func_async(self, func):
        """Async method for tool execution."""
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


class TestRunnerTracing(unittest.TestCase):
    """Test Runner tracing functionality."""

    def setUp(self):
        """Set up test environment."""
        # Store original state
        self.original_disabled = GLOBAL_TRACE_PROVIDER._disabled
        self.original_processors = list(
            GLOBAL_TRACE_PROVIDER._multi_processor._processors
        )

        # Reset to clean state
        set_tracing_disabled(False)
        set_trace_processors([])

        # Set up mock processor
        self.processor = MockTracingProcessor()
        add_trace_processor(self.processor)

    def tearDown(self):
        """Clean up test environment."""
        # Restore original state
        GLOBAL_TRACE_PROVIDER._disabled = self.original_disabled
        GLOBAL_TRACE_PROVIDER._multi_processor.set_processors(self.original_processors)

    def test_runner_call_creates_spans(self):
        """Test that Runner.call creates proper spans."""
        fn = DummyFunction(name="finish", _is_answer_final=True, _answer="done")
        mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(output="done")
        agent = DummyAgent(
            planner=FakePlanner([GeneratorOutput(data=fn)]),
            answer_data_type=None,
            tool_manager=mock_tool_manager,
        )
        runner = Runner(agent=agent)

        # No trace exists, so spans should be NoOp
        result = runner.call(prompt_kwargs={"query": "test"})

        # Should still return valid result
        self.assertIsInstance(result, RunnerResult)
        self.assertEqual(result.answer, "done")

        # Since no trace was active, spans should be NoOp and not recorded
        self.assertEqual(len(self.processor.span_starts), 0)

    def test_runner_call_with_trace_creates_spans(self):
        """Test that Runner.call with active trace creates proper spans."""
        fn = DummyFunction(name="finish", _is_answer_final=True, _answer="done")
        mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(output="done")
        agent = DummyAgent(
            planner=FakePlanner([GeneratorOutput(data=fn)]),
            answer_data_type=None,
            tool_manager=mock_tool_manager,
        )
        runner = Runner(agent=agent)

        with trace("test_workflow"):
            result = runner.call(prompt_kwargs={"query": "test"})

        # Should still return valid result
        self.assertIsInstance(result, RunnerResult)
        self.assertEqual(result.answer, "done")

        # Should have created multiple spans
        self.assertGreater(len(self.processor.span_starts), 0)

        # Check for runner span
        runner_spans = [
            span
            for span in self.processor.span_starts
            if isinstance(span.span_data, AdalFlowRunnerSpanData)
        ]
        self.assertEqual(len(runner_spans), 1)
        runner_span = runner_spans[0]
        self.assertEqual(runner_span.span_data.workflow_status, "completed")
        self.assertEqual(runner_span.span_data.steps_executed, 1)

        # Check for generator span
        generator_spans = [
            span
            for span in self.processor.span_starts
            if isinstance(span.span_data, AdalFlowGeneratorSpanData)
        ]
        self.assertEqual(len(generator_spans), 1)
        generator_span = generator_spans[0]
        self.assertEqual(generator_span.span_data.generator_id, "fake_planner")

        # When _is_answer_final=True, no tool spans are created since tools aren't executed
        tool_spans = [
            span
            for span in self.processor.span_starts
            if isinstance(span.span_data, AdalFlowToolSpanData)
        ]
        self.assertEqual(len(tool_spans), 0)  # No tool execution for final answers

        # Check for step span
        step_spans = [
            span
            for span in self.processor.span_starts
            if isinstance(span.span_data, AdalFlowStepSpanData)
        ]
        self.assertEqual(len(step_spans), 1)
        step_span = step_spans[0]
        self.assertEqual(step_span.span_data.step_number, 0)
        self.assertEqual(step_span.span_data.action_type, "planning")
        self.assertIsNone(
            step_span.span_data.tool_name
        )  # No tool execution for final answers
        self.assertFalse(step_span.span_data.is_final)  # Step span not marked as final

        # Check for response span
        response_spans = [
            span
            for span in self.processor.span_starts
            if isinstance(span.span_data, AdalFlowResponseSpanData)
        ]
        self.assertEqual(len(response_spans), 1)
        response_span = response_spans[0]
        self.assertEqual(response_span.span_data.answer, "done")
        self.assertEqual(response_span.span_data.result_type, "str")

    def test_runner_acall_creates_spans(self):
        """Test that Runner.acall creates proper spans."""

        async def async_test():
            fn = DummyFunction(
                name="finish", _is_answer_final=True, _answer="async_done"
            )
            mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(
                output="async_done"
            )
            agent = DummyAgent(
                planner=FakePlanner([GeneratorOutput(data=fn)]),
                answer_data_type=None,
                tool_manager=mock_tool_manager,
            )
            runner = Runner(agent=agent)

            with trace("test_workflow"):
                result = await runner.acall(prompt_kwargs={"query": "test"})

            # Should return valid result
            self.assertIsInstance(result, RunnerResult)
            self.assertEqual(result.answer, "async_done")

            # Should have created multiple spans
            self.assertGreater(len(self.processor.span_starts), 0)

            # Check for runner span
            runner_spans = [
                span
                for span in self.processor.span_starts
                if isinstance(span.span_data, AdalFlowRunnerSpanData)
            ]
            self.assertEqual(len(runner_spans), 1)
            runner_span = runner_spans[0]
            self.assertEqual(runner_span.span_data.workflow_status, "completed")

            # Check for generator span with async planner
            generator_spans = [
                span
                for span in self.processor.span_starts
                if isinstance(span.span_data, AdalFlowGeneratorSpanData)
            ]
            self.assertEqual(len(generator_spans), 1)
            generator_span = generator_spans[0]
            self.assertEqual(generator_span.span_data.generator_id, "async_planner")

        asyncio.run(async_test())

    def test_runner_astream_creates_spans(self):
        """Test that Runner.astream creates proper spans for streaming execution."""

        async def async_test():
            fn = DummyFunction(
                name="finish", _is_answer_final=True, _answer="stream_done"
            )
            mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(
                output="stream_done"
            )
            agent = DummyAgent(
                planner=FakePlanner([GeneratorOutput(data=fn)]),
                answer_data_type=None,
                tool_manager=mock_tool_manager,
            )
            runner = Runner(agent=agent)

            with trace("test_stream_workflow"):
                stream_result = runner.astream(prompt_kwargs={"query": "test"})

                # Wait for streaming to complete
                await stream_result.wait_for_completion()

            # Should have valid result in stream_result properties
            self.assertEqual(stream_result.answer, "stream_done")

            print("stream_result:", stream_result)
            self.assertIsNotNone(stream_result.step_history)

            # Should have created multiple spans
            self.assertGreater(len(self.processor.span_starts), 0)

            # Check for runner span with streaming workflow status
            runner_spans = [
                span
                for span in self.processor.span_starts
                if isinstance(span.span_data, AdalFlowRunnerSpanData)
            ]
            self.assertEqual(len(runner_spans), 1)
            runner_span = runner_spans[0]
            self.assertEqual(runner_span.span_data.workflow_status, "stream_completed", msg=f"span data: {stream_result}")

            # Check for generator span with stream planner
            generator_spans = [
                span
                for span in self.processor.span_starts
                if isinstance(span.span_data, AdalFlowGeneratorSpanData)
            ]
            self.assertEqual(len(generator_spans), 1)
            generator_span = generator_spans[0]
            self.assertEqual(generator_span.span_data.generator_id, "async_planner")

            # Check for step span with streaming action type
            step_spans = [
                span
                for span in self.processor.span_starts
                if isinstance(span.span_data, AdalFlowStepSpanData)
            ]
            self.assertEqual(len(step_spans), 1)
            step_span = step_spans[0]
            self.assertEqual(step_span.span_data.action_type, "stream_planning")
            self.assertFalse(
                step_span.span_data.is_final
            )  # Step span is not marked as final in streaming case

            # Check for response span with streaming metadata
            response_spans = [
                span
                for span in self.processor.span_starts
                if isinstance(span.span_data, AdalFlowResponseSpanData)
            ]
            self.assertEqual(len(response_spans), 1)
            response_span = response_spans[0]
            self.assertEqual(response_span.span_data.answer, "stream_done")
            self.assertTrue(response_span.span_data.execution_metadata.get("streaming"))

        asyncio.run(async_test())

    def test_runner_astream_multi_step_spans(self):
        """Test Runner.astream with multiple steps creates proper spans."""

        async def async_test():
            fn1 = DummyFunction(name="search", _is_answer_final=False)
            fn2 = DummyFunction(
                name="finish", _is_answer_final=True, _answer="final_stream_result"
            )
            mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(
                output=(
                    "search_result"
                    if expr_or_fun.name == "search"
                    else "final_stream_result"
                )
            )
            agent = DummyAgent(
                planner=FakePlanner(
                    [GeneratorOutput(data=fn1), GeneratorOutput(data=fn2)]
                ),
                answer_data_type=None,
                tool_manager=mock_tool_manager,
            )
            runner = Runner(agent=agent)

            with trace("test_stream_workflow"):
                stream_result = runner.astream(prompt_kwargs={"query": "test"})

                # Wait for streaming to complete
                await stream_result.wait_for_completion()

            # Should have valid result in stream_result properties
            self.assertEqual(stream_result.answer, "final_stream_result")
            self.assertIsNotNone(stream_result.step_history)

            # Should have created spans for both steps
            step_spans = [
                span
                for span in self.processor.span_starts
                if isinstance(span.span_data, AdalFlowStepSpanData)
            ]
            self.assertEqual(len(step_spans), 2)

            # Check first step
            first_step = step_spans[0]
            self.assertEqual(first_step.span_data.step_number, 0)
            self.assertEqual(first_step.span_data.tool_name, "search")
            self.assertEqual(first_step.span_data.action_type, "stream_planning")
            self.assertFalse(first_step.span_data.is_final)

            # Check second step
            second_step = step_spans[1]
            self.assertEqual(second_step.span_data.step_number, 1)
            self.assertIsNone(
                second_step.span_data.tool_name
            )  # No tool execution for final answers
            self.assertEqual(second_step.span_data.action_type, "stream_planning")
            self.assertFalse(
                second_step.span_data.is_final
            )  # Step span is not marked as final in streaming case

            # Check runner span shows correct final state for streaming
            runner_spans = [
                span
                for span in self.processor.span_starts
                if isinstance(span.span_data, AdalFlowRunnerSpanData)
            ]
            self.assertEqual(len(runner_spans), 1)
            runner_span = runner_spans[0]
            self.assertEqual(runner_span.span_data.steps_executed, 2)
            self.assertEqual(runner_span.span_data.final_answer, "final_stream_result")
            self.assertEqual(runner_span.span_data.workflow_status, "stream_completed")

        asyncio.run(async_test())

    def test_runner_astream_error_handling_spans(self):
        """Test Runner.astream error handling creates proper spans."""

        async def async_test():
            fn = DummyFunction(name="error_function", _is_answer_final=False)

            def error_tool_manager(expr_or_fun, step):
                _ = expr_or_fun, step  # Suppress unused variable warnings
                raise ValueError("Stream test error")

            agent = DummyAgent(
                planner=FakePlanner([GeneratorOutput(data=fn)]),
                answer_data_type=None,
                tool_manager=error_tool_manager,
            )
            runner = Runner(agent=agent)

            with trace("test_stream_workflow"):
                stream_result = runner.astream(prompt_kwargs={"query": "test"})

                # Wait for streaming to complete (with error)
                await stream_result.wait_for_completion()

            # Should have error in streaming result
            self.assertIsNotNone(stream_result._exception)
            self.assertTrue(stream_result._exception.startswith("Error in step 0:"))

            # Should have created spans despite error
            runner_spans = [
                span
                for span in self.processor.span_starts
                if isinstance(span.span_data, AdalFlowRunnerSpanData)
            ]
            self.assertEqual(len(runner_spans), 1)
            runner_span = runner_spans[0]
            self.assertEqual(runner_span.span_data.workflow_status, "stream_failed")
            # Streaming stops after error with incomplete result - but there's no final_answer when it fails
            self.assertIsNone(runner_span.span_data.final_answer)

            # Should have at least one response span
            response_spans = [
                span
                for span in self.processor.span_starts
                if isinstance(span.span_data, AdalFlowResponseSpanData)
            ]
            self.assertGreaterEqual(len(response_spans), 1)

            # Check the response span has streaming metadata and appropriate workflow status
            response_span = response_spans[0]
            self.assertTrue(
                response_span.span_data.execution_metadata.get("streaming")
            )
            self.assertEqual(response_span.span_data.execution_metadata.get("workflow_status"), "stream_failed")

        asyncio.run(async_test())

    def test_runner_multi_step_spans(self):
        """Test Runner with multiple steps creates proper spans."""
        fn1 = DummyFunction(name="search", _is_answer_final=False)
        fn2 = DummyFunction(
            name="finish", _is_answer_final=True, _answer="final_result"
        )
        mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(
            output="search_result" if expr_or_fun.name == "search" else "final_result"
        )
        agent = DummyAgent(
            planner=FakePlanner([GeneratorOutput(data=fn1), GeneratorOutput(data=fn2)]),
            answer_data_type=None,
            tool_manager=mock_tool_manager,
        )
        runner = Runner(agent=agent)

        with trace("test_workflow"):
            result = runner.call(prompt_kwargs={"query": "test"})

        # Should return valid result
        self.assertIsInstance(result, RunnerResult)
        self.assertEqual(result.answer, "final_result")

        # Should have created spans for both steps
        step_spans = [
            span
            for span in self.processor.span_starts
            if isinstance(span.span_data, AdalFlowStepSpanData)
        ]
        self.assertEqual(len(step_spans), 2)

        # Check first step
        first_step = step_spans[0]
        self.assertEqual(first_step.span_data.step_number, 0)
        self.assertEqual(first_step.span_data.tool_name, "search")
        self.assertFalse(first_step.span_data.is_final)

        # Check second step
        second_step = step_spans[1]
        self.assertEqual(second_step.span_data.step_number, 1)
        self.assertIsNone(
            second_step.span_data.tool_name
        )  # No tool execution for final answers
        self.assertFalse(
            second_step.span_data.is_final
        )  # Step span not marked as final

        # Should have multiple generator spans (one per step)
        generator_spans = [
            span
            for span in self.processor.span_starts
            if isinstance(span.span_data, AdalFlowGeneratorSpanData)
        ]
        self.assertEqual(len(generator_spans), 2)

        # Should have tool spans only for non-final steps (first step only)
        tool_spans = [
            span
            for span in self.processor.span_starts
            if isinstance(span.span_data, AdalFlowToolSpanData)
        ]
        self.assertEqual(len(tool_spans), 1)  # Only first step executes tools

        # Check runner span shows correct final state
        runner_spans = [
            span
            for span in self.processor.span_starts
            if isinstance(span.span_data, AdalFlowRunnerSpanData)
        ]
        self.assertEqual(len(runner_spans), 1)
        runner_span = runner_spans[0]
        self.assertEqual(runner_span.span_data.steps_executed, 2)
        self.assertEqual(runner_span.span_data.final_answer, "final_result")
        self.assertEqual(runner_span.span_data.workflow_status, "completed")

    def test_runner_error_handling_spans(self):
        """Test Runner error handling creates proper spans."""
        fn = DummyFunction(name="error_function", _is_answer_final=False)

        def error_tool_manager(expr_or_fun, step):
            _ = expr_or_fun, step  # Suppress unused variable warnings
            raise ValueError("Test error")

        agent = DummyAgent(
            planner=FakePlanner([GeneratorOutput(data=fn)]),
            answer_data_type=None,
            tool_manager=error_tool_manager,
        )
        runner = Runner(agent=agent)

        with trace("test_workflow"):
            result = runner.call(prompt_kwargs={"query": "test"})

        # Should return incomplete result after error (runner stops after error)
        self.assertIsInstance(result, RunnerResult)
        self.assertIsNotNone(result.error)  # Error is now captured in the result
        self.assertEqual(result.answer, "Error in step 0: Test error")  # Error message as answer

        # Should have created spans for error and incomplete result
        runner_spans = [
            span
            for span in self.processor.span_starts
            if isinstance(span.span_data, AdalFlowRunnerSpanData)
        ]
        self.assertEqual(len(runner_spans), 1)
        runner_span = runner_spans[0]
        self.assertEqual(
            runner_span.span_data.workflow_status, "completed"
        )  # Final status is completed
        self.assertIsNone(
            runner_span.span_data.final_answer
        )  # No final answer when no output generated

        # Should have multiple response spans (error + incomplete result)
        response_spans = [
            span
            for span in self.processor.span_starts
            if isinstance(span.span_data, AdalFlowResponseSpanData)
        ]
        self.assertEqual(len(response_spans), 2)  # One for error, one for final result

        # Check error response span (first one)
        error_response_span = response_spans[0]
        self.assertEqual(error_response_span.span_data.result_type, "error")
        self.assertIn("Error in step 0:", error_response_span.span_data.answer)

        # Check final response span (last one)
        final_response_span = response_spans[1]
        self.assertEqual(final_response_span.span_data.result_type, "no_output")
        self.assertEqual(final_response_span.span_data.answer, "No output generated after 1 steps (max_steps: 10)")

    def test_runner_disabled_tracing(self):
        """Test Runner with disabled tracing."""
        set_tracing_disabled(True)

        fn = DummyFunction(name="finish", _is_answer_final=True, _answer="done")
        mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(output="done")
        agent = DummyAgent(
            planner=FakePlanner([GeneratorOutput(data=fn)]),
            answer_data_type=None,
            tool_manager=mock_tool_manager,
        )
        runner = Runner(agent=agent)

        with trace("test_workflow"):
            result = runner.call(prompt_kwargs={"query": "test"})

        # Should still return valid result
        self.assertIsInstance(result, RunnerResult)
        self.assertEqual(result.answer, "done")

        # Should not have created any spans
        self.assertEqual(len(self.processor.span_starts), 0)
        self.assertEqual(len(self.processor.span_ends), 0)

    def test_runner_environment_variable_disable(self):
        """Test Runner with ADALFLOW_DISABLE_TRACING environment variable."""
        # Test with environment variable set
        with patch.dict(os.environ, {"ADALFLOW_DISABLE_TRACING": "true"}):
            # Reset provider to pick up env var
            from adalflow.tracing.setup import TraceProvider

            test_provider = TraceProvider()

            # Temporarily replace global provider
            original_provider = GLOBAL_TRACE_PROVIDER
            import adalflow.tracing.setup
            import adalflow.tracing.create

            adalflow.tracing.setup.GLOBAL_TRACE_PROVIDER = test_provider
            adalflow.tracing.create.GLOBAL_TRACE_PROVIDER = test_provider

            try:
                fn = DummyFunction(name="finish", _is_answer_final=True, _answer="done")
                mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(
                    output="done"
                )
                agent = DummyAgent(
                    planner=FakePlanner([GeneratorOutput(data=fn)]),
                    answer_data_type=None,
                    tool_manager=mock_tool_manager,
                )
                runner = Runner(agent=agent)

                with trace("test_workflow"):
                    result = runner.call(prompt_kwargs={"query": "test"})

                # Should still return valid result
                self.assertIsInstance(result, RunnerResult)
                self.assertEqual(result.answer, "done")

                # Should not have created any spans (processor is on original provider)
                self.assertEqual(len(self.processor.span_starts), 0)

            finally:
                # Restore original provider
                adalflow.tracing.setup.GLOBAL_TRACE_PROVIDER = original_provider
                adalflow.tracing.create.GLOBAL_TRACE_PROVIDER = original_provider

    def test_runner_span_data_updates(self):
        """Test that runner span data gets updated correctly during execution."""
        fn = DummyFunction(name="finish", _is_answer_final=True, _answer="done")
        mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(output="done")
        agent = DummyAgent(
            planner=FakePlanner([GeneratorOutput(data=fn)]),
            answer_data_type=None,
            tool_manager=mock_tool_manager,
        )
        runner = Runner(agent=agent)

        with trace("test_workflow"):
            _result = runner.call(prompt_kwargs={"query": "test"})

        # Get runner span
        runner_spans = [
            span
            for span in self.processor.span_starts
            if isinstance(span.span_data, AdalFlowRunnerSpanData)
        ]
        self.assertEqual(len(runner_spans), 1)
        runner_span = runner_spans[0]

        # Check that span data was updated correctly
        self.assertEqual(runner_span.span_data.steps_executed, 1)
        self.assertEqual(runner_span.span_data.final_answer, "done")
        self.assertEqual(runner_span.span_data.workflow_status, "completed")

        # Check that data dict was also updated (for MLflow compatibility)
        self.assertEqual(runner_span.span_data.data["steps_executed"], 1)
        self.assertEqual(runner_span.span_data.data["final_answer"], "done")
        self.assertEqual(runner_span.span_data.data["workflow_status"], "completed")

    def test_runner_max_steps_without_finish(self):
        """Test Runner that reaches max steps without finish."""
        # Create functions that don't finish
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

        with trace("test_workflow"):
            result = runner.call(prompt_kwargs={"query": "test"})

        # Should return RunnerResult with "No output generated" message when max steps reached without finish
        self.assertIsInstance(result, RunnerResult)
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.answer)
        self.assertTrue(result.answer.startswith("No output generated"))

        # Should have created spans for 3 steps
        step_spans = [
            span
            for span in self.processor.span_starts
            if isinstance(span.span_data, AdalFlowStepSpanData)
        ]
        self.assertEqual(len(step_spans), 3)

        # Check runner span shows incomplete status
        runner_spans = [
            span
            for span in self.processor.span_starts
            if isinstance(span.span_data, AdalFlowRunnerSpanData)
        ]
        self.assertEqual(len(runner_spans), 1)
        runner_span = runner_spans[0]
        self.assertEqual(runner_span.span_data.steps_executed, 3)
        self.assertEqual(runner_span.span_data.workflow_status, "completed")

        # Should have response span for incomplete workflow
        response_spans = [
            span
            for span in self.processor.span_starts
            if isinstance(span.span_data, AdalFlowResponseSpanData)
        ]
        self.assertEqual(len(response_spans), 1)
        response_span = response_spans[0]
        self.assertEqual(response_span.span_data.result_type, "no_output")
        self.assertIn("No output generated", response_span.span_data.answer)

    def test_runner_span_hierarchy(self):
        """Test that runner spans have proper hierarchy."""
        fn = DummyFunction(name="finish", _is_answer_final=True, _answer="done")
        mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(output="done")
        agent = DummyAgent(
            planner=FakePlanner([GeneratorOutput(data=fn)]),
            answer_data_type=None,
            tool_manager=mock_tool_manager,
        )
        runner = Runner(agent=agent)

        with trace("test_workflow") as trace_obj:
            _result = runner.call(prompt_kwargs={"query": "test"})

        # All spans should be children of the trace
        for span in self.processor.span_starts:
            self.assertEqual(span.trace_id, trace_obj.trace_id)

        # Check that step spans have runner span as parent
        runner_spans = [
            span
            for span in self.processor.span_starts
            if isinstance(span.span_data, AdalFlowRunnerSpanData)
        ]
        step_spans = [
            span
            for span in self.processor.span_starts
            if isinstance(span.span_data, AdalFlowStepSpanData)
        ]

        if runner_spans and step_spans:
            runner_span = runner_spans[0]
            step_span = step_spans[0]
            self.assertEqual(step_span.parent_id, runner_span.span_id)


class TestRunnerTracingPerformance(unittest.TestCase):
    """Test Runner tracing performance."""

    def setUp(self):
        """Set up test environment."""
        self.original_disabled = GLOBAL_TRACE_PROVIDER._disabled
        set_tracing_disabled(False)
        set_trace_processors([])

    def tearDown(self):
        """Clean up test environment."""
        GLOBAL_TRACE_PROVIDER._disabled = self.original_disabled

    def test_runner_performance_with_disabled_tracing(self):
        """Test Runner performance with disabled tracing."""
        set_tracing_disabled(True)

        fn = DummyFunction(name="finish", _is_answer_final=True, _answer="done")
        mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(output="done")
        # Create enough outputs for 10 iterations
        outputs = [GeneratorOutput(data=fn) for _ in range(10)]
        agent = DummyAgent(
            planner=FakePlanner(outputs),
            answer_data_type=None,
            tool_manager=mock_tool_manager,
        )
        runner = Runner(agent=agent)

        import time

        start_time = time.time()

        # Run multiple times to test performance
        for i in range(10):
            result = runner.call(prompt_kwargs={"query": f"test_{i}"})
            self.assertIsInstance(result, RunnerResult)
            self.assertEqual(result.answer, "done")

        end_time = time.time()
        duration = end_time - start_time

        # Should complete reasonably quickly (less than 5 seconds for 10 runs)
        self.assertLess(duration, 5.0)

    def test_runner_performance_with_many_spans(self):
        """Test Runner performance with many spans."""
        processor = MockTracingProcessor()
        add_trace_processor(processor)

        # Create many steps
        functions = [
            DummyFunction(name=f"step_{i}", _is_answer_final=False) for i in range(50)
        ]
        functions.append(
            DummyFunction(name="finish", _is_answer_final=True, _answer="final_result")
        )
        outputs = [GeneratorOutput(data=fn) for fn in functions]

        mock_tool_manager = lambda expr_or_fun, step: SimpleNamespace(
            output="final_result" if expr_or_fun.name == "finish" else "intermediate"
        )
        agent = DummyAgent(
            planner=FakePlanner(outputs),
            max_steps=51,
            answer_data_type=None,
            tool_manager=mock_tool_manager,
        )
        runner = Runner(agent=agent)

        import time

        start_time = time.time()

        with trace("test_workflow"):
            result = runner.call(prompt_kwargs={"query": "test"})

        end_time = time.time()
        duration = end_time - start_time

        # Should complete in reasonable time (less than 10 seconds)
        self.assertLess(duration, 10.0)

        # Should have created all spans
        step_spans = [
            span
            for span in processor.span_starts
            if isinstance(span.span_data, AdalFlowStepSpanData)
        ]
        self.assertEqual(len(step_spans), 51)  # 50 steps + 1 finish

        # Should return valid result
        self.assertIsInstance(result, RunnerResult)
        self.assertEqual(result.answer, "final_result")


if __name__ == "__main__":
    unittest.main()
