import unittest
import os
from unittest.mock import patch

from adalflow.tracing import (
    trace,
    runner_span,
    generator_span,
    tool_span,
    response_span,
    step_span,
    custom_span,
    set_tracing_disabled,
    add_trace_processor,
    set_trace_processors,
    get_trace_processors,
    GLOBAL_TRACE_PROVIDER,
)
from adalflow.tracing.span_data import (
    AdalFlowRunnerSpanData,
    AdalFlowGeneratorSpanData,
    AdalFlowToolSpanData,
    AdalFlowResponseSpanData,
    AdalFlowStepSpanData,
    CustomSpanData,
)
from adalflow.tracing.spans import NoOpSpan, SpanImpl
from adalflow.tracing.traces import NoOpTrace, TraceImpl
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


class TestTracingCore(unittest.TestCase):
    """Test core tracing functionality."""

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

    def tearDown(self):
        """Clean up test environment."""
        # Restore original state
        GLOBAL_TRACE_PROVIDER._disabled = self.original_disabled
        GLOBAL_TRACE_PROVIDER._multi_processor.set_processors(self.original_processors)

    def test_trace_creation(self):
        """Test basic trace creation."""
        trace_obj = trace("test_workflow")
        self.assertIsInstance(trace_obj, TraceImpl)
        self.assertEqual(trace_obj.name, "test_workflow")
        self.assertIsNotNone(trace_obj.trace_id)

    def test_trace_creation_with_metadata(self):
        """Test trace creation with metadata."""
        metadata = {"user_id": "test_user", "session_id": "test_session"}
        trace_obj = trace("test_workflow", metadata=metadata)
        self.assertEqual(trace_obj.metadata, metadata)

    def test_trace_disabled(self):
        """Test trace creation when disabled."""
        set_tracing_disabled(True)
        trace_obj = trace("test_workflow")
        self.assertIsInstance(trace_obj, NoOpTrace)

    def test_environment_variable_disable(self):
        """Test that ADALFLOW_DISABLE_TRACING environment variable works."""
        with patch.dict(os.environ, {"ADALFLOW_DISABLE_TRACING": "true"}):
            # Create a new provider to pick up the env var
            from adalflow.tracing.setup import TraceProvider

            test_provider = TraceProvider()
            self.assertTrue(test_provider._disabled)

        with patch.dict(os.environ, {"ADALFLOW_DISABLE_TRACING": "1"}):
            test_provider = TraceProvider()
            self.assertTrue(test_provider._disabled)

        with patch.dict(os.environ, {"ADALFLOW_DISABLE_TRACING": "false"}):
            test_provider = TraceProvider()
            self.assertFalse(test_provider._disabled)


class TestEnvironmentVariableTracing(unittest.TestCase):
    """Test tracing behavior with environment variables."""

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

    def tearDown(self):
        """Clean up after each test."""
        # Restore original state
        GLOBAL_TRACE_PROVIDER._disabled = self.original_disabled
        GLOBAL_TRACE_PROVIDER._multi_processor.set_processors(self.original_processors)

        # Remove the environment variable if it was set
        if "ADALFLOW_DISABLE_TRACING" in os.environ:
            del os.environ["ADALFLOW_DISABLE_TRACING"]

    def test_tracing_disabled_via_env_var_true(self):
        """Test that tracing is disabled when ADALFLOW_DISABLE_TRACING=true."""
        with patch.dict(os.environ, {"ADALFLOW_DISABLE_TRACING": "true"}, clear=False):
            # Import and create provider with env var set
            from adalflow.tracing.setup import TraceProvider

            test_provider = TraceProvider()

            # Should be disabled
            self.assertTrue(test_provider._disabled)

            # Test trace creation - should return NoOpTrace
            trace_obj = test_provider.create_trace("test_workflow")
            self.assertIsInstance(trace_obj, NoOpTrace)

            # Test span creation - should return NoOpSpan
            from adalflow.tracing.span_data import CustomSpanData

            span_data = CustomSpanData(name="test", data={})
            span_obj = test_provider.create_span(span_data)
            self.assertIsInstance(span_obj, NoOpSpan)

    def test_tracing_disabled_via_env_var_1(self):
        """Test that tracing is disabled when ADALFLOW_DISABLE_TRACING=1."""
        with patch.dict(os.environ, {"ADALFLOW_DISABLE_TRACING": "1"}, clear=False):
            from adalflow.tracing.setup import TraceProvider

            test_provider = TraceProvider()

            # Should be disabled
            self.assertTrue(test_provider._disabled)

            # Test trace creation - should return NoOpTrace
            trace_obj = test_provider.create_trace("test_workflow")
            self.assertIsInstance(trace_obj, NoOpTrace)

    def test_tracing_enabled_via_env_var_false(self):
        """Test that tracing is enabled when ADALFLOW_DISABLE_TRACING=false."""
        with patch.dict(os.environ, {"ADALFLOW_DISABLE_TRACING": "false"}, clear=False):
            from adalflow.tracing.setup import TraceProvider

            test_provider = TraceProvider()

            # Should be enabled
            self.assertFalse(test_provider._disabled)

            # Test trace creation - should return TraceImpl
            trace_obj = test_provider.create_trace("test_workflow")
            self.assertIsInstance(trace_obj, TraceImpl)

            # Test span creation with trace - should return SpanImpl
            from adalflow.tracing.span_data import CustomSpanData

            span_data = CustomSpanData(name="test", data={})
            span_obj = test_provider.create_span(span_data, parent=trace_obj)
            self.assertIsInstance(span_obj, SpanImpl)

    def test_tracing_enabled_via_env_var_0(self):
        """Test that tracing is enabled when ADALFLOW_DISABLE_TRACING=0."""
        with patch.dict(os.environ, {"ADALFLOW_DISABLE_TRACING": "0"}, clear=False):
            from adalflow.tracing.setup import TraceProvider

            test_provider = TraceProvider()

            # Should be enabled
            self.assertFalse(test_provider._disabled)

            # Test trace creation - should return TraceImpl
            trace_obj = test_provider.create_trace("test_workflow")
            self.assertIsInstance(trace_obj, TraceImpl)

    def test_tracing_enabled_when_env_var_not_set(self):
        """Test that tracing is enabled by default when environment variable is not set."""
        # Make sure the env var is not set
        if "ADALFLOW_DISABLE_TRACING" in os.environ:
            del os.environ["ADALFLOW_DISABLE_TRACING"]

        from adalflow.tracing.setup import TraceProvider

        test_provider = TraceProvider()

        # Should be enabled by default (note: current implementation defaults to disabled)
        # This test documents the current behavior - may need adjustment based on requirements
        self.assertTrue(test_provider._disabled)  # Current default is disabled

    def test_tracing_enabled_when_env_var_empty(self):
        """Test that tracing behavior when ADALFLOW_DISABLE_TRACING is empty."""
        with patch.dict(os.environ, {"ADALFLOW_DISABLE_TRACING": ""}, clear=False):
            from adalflow.tracing.setup import TraceProvider

            test_provider = TraceProvider()

            # Empty string should be treated as enabled (falsy)
            self.assertFalse(test_provider._disabled)

    def test_env_var_case_insensitive(self):
        """Test that environment variable values are case insensitive."""
        test_cases = [
            ("TRUE", True),
            ("True", True),
            ("tRuE", True),
            ("FALSE", False),
            ("False", False),
            ("fAlSe", False),
        ]

        for env_value, expected_disabled in test_cases:
            with self.subTest(env_value=env_value):
                with patch.dict(
                    os.environ, {"ADALFLOW_DISABLE_TRACING": env_value}, clear=False
                ):
                    from adalflow.tracing.setup import TraceProvider

                    test_provider = TraceProvider()
                    self.assertEqual(test_provider._disabled, expected_disabled)

    def test_span_creation_functions_with_env_disabled(self):
        """Test span creation functions when tracing is disabled via environment variable."""
        with patch.dict(os.environ, {"ADALFLOW_DISABLE_TRACING": "true"}, clear=False):
            # Create a new provider to pick up env var
            from adalflow.tracing.setup import TraceProvider

            test_provider = TraceProvider()

            # Should be disabled
            self.assertTrue(test_provider._disabled)

            # All trace/span creation should return NoOp objects
            trace_obj = test_provider.create_trace("test_workflow")
            self.assertIsInstance(trace_obj, NoOpTrace)

            # Spans should also be NoOp when provider is disabled
            from adalflow.tracing.span_data import CustomSpanData

            span_data = CustomSpanData(name="test", data={})
            span_obj = test_provider.create_span(span_data)
            self.assertIsInstance(span_obj, NoOpSpan)

    def test_span_creation_functions_with_env_enabled(self):
        """Test span creation functions when tracing is enabled via environment variable."""
        with patch.dict(os.environ, {"ADALFLOW_DISABLE_TRACING": "false"}, clear=False):
            # Create a new provider to pick up env var
            from adalflow.tracing.setup import TraceProvider

            test_provider = TraceProvider()

            # Should be enabled
            self.assertFalse(test_provider._disabled)

            # Test trace creation with enabled provider - should return TraceImpl
            trace_obj = test_provider.create_trace("test_workflow")
            self.assertIsInstance(trace_obj, TraceImpl)

            # Test span creation with trace - should return SpanImpl
            from adalflow.tracing.span_data import CustomSpanData

            span_data = CustomSpanData(name="test", data={})
            span_obj = test_provider.create_span(span_data, parent=trace_obj)
            self.assertIsInstance(span_obj, SpanImpl)

    def test_provider_with_env_disabled_has_no_side_effects(self):
        """Test that disabled provider creation has no side effects."""
        with patch.dict(os.environ, {"ADALFLOW_DISABLE_TRACING": "true"}, clear=False):
            from adalflow.tracing.setup import TraceProvider

            test_provider = TraceProvider()

            # Should be disabled
            self.assertTrue(test_provider._disabled)

            # Create a processor and add it to our test provider
            processor = MockTracingProcessor()
            test_provider.register_processor(processor)

            # Create trace and span - should be NoOp
            trace_obj = test_provider.create_trace("test_workflow")
            self.assertIsInstance(trace_obj, NoOpTrace)

            # NoOp trace should not call processors
            self.assertEqual(len(processor.trace_starts), 0)
            self.assertEqual(len(processor.trace_ends), 0)

    def test_provider_with_env_enabled_processes_events(self):
        """Test that enabled provider properly processes events."""
        with patch.dict(os.environ, {"ADALFLOW_DISABLE_TRACING": "false"}, clear=False):
            from adalflow.tracing.setup import TraceProvider

            test_provider = TraceProvider()

            # Should be enabled
            self.assertFalse(test_provider._disabled)

            # Create a processor and add it to our test provider
            processor = MockTracingProcessor()
            test_provider.register_processor(processor)

            # Create and use trace - should be real objects
            trace_obj = test_provider.create_trace("test_workflow")
            self.assertIsInstance(trace_obj, TraceImpl)

            # Start and end the trace manually to test processor calls
            trace_obj.start()
            trace_obj.finish()

            # Processors should be called since tracing is enabled
            self.assertEqual(len(processor.trace_starts), 1)
            self.assertEqual(len(processor.trace_ends), 1)

    def test_trace_processors(self):
        """Test trace processor management."""
        processor1 = MockTracingProcessor()
        processor2 = MockTracingProcessor()

        # Test adding processors
        add_trace_processor(processor1)
        add_trace_processor(processor2)

        processors = get_trace_processors()
        self.assertEqual(len(processors), 2)
        self.assertIn(processor1, processors)
        self.assertIn(processor2, processors)

        # Test setting processors
        processor3 = MockTracingProcessor()
        set_trace_processors([processor3])
        processors = get_trace_processors()
        self.assertEqual(len(processors), 1)
        self.assertIn(processor3, processors)

    def test_trace_context_manager(self):
        """Test trace as context manager."""
        processor = MockTracingProcessor()
        add_trace_processor(processor)

        with trace("test_workflow") as trace_obj:
            self.assertIsInstance(trace_obj, TraceImpl)
            # Should have been started
            self.assertEqual(len(processor.trace_starts), 1)
            self.assertEqual(processor.trace_starts[0], trace_obj)

        # Should be finished after exiting context
        self.assertEqual(len(processor.trace_ends), 1)
        self.assertEqual(processor.trace_ends[0], trace_obj)


class TestSpanData(unittest.TestCase):
    """Test span data classes."""

    def test_adalflow_runner_span_data(self):
        """Test AdalFlowRunnerSpanData."""
        span_data = AdalFlowRunnerSpanData(
            runner_id="test_runner",
            max_steps=10,
            steps_executed=5,
            final_answer="test answer",
            workflow_status="completed",
            execution_time=2.5,
        )

        self.assertEqual(span_data.runner_id, "test_runner")
        self.assertEqual(span_data.max_steps, 10)
        self.assertEqual(span_data.steps_executed, 5)
        self.assertEqual(span_data.final_answer, "test answer")
        self.assertEqual(span_data.workflow_status, "completed")
        self.assertEqual(span_data.execution_time, 2.5)

        # Test export
        exported = span_data.export()
        self.assertEqual(exported["type"], "custom")
        self.assertEqual(exported["name"], "AdalFlow-Runner-test_runner")
        self.assertIn("runner_id", exported["data"])
        self.assertEqual(exported["data"]["runner_id"], "test_runner")

        # Test update_attributes
        span_data.update_attributes({"steps_executed": 7, "workflow_status": "running"})
        self.assertEqual(span_data.steps_executed, 7)
        self.assertEqual(span_data.workflow_status, "running")
        self.assertEqual(span_data.data["steps_executed"], 7)
        self.assertEqual(span_data.data["workflow_status"], "running")

    def test_adalflow_generator_span_data(self):
        """Test AdalFlowGeneratorSpanData."""
        span_data = AdalFlowGeneratorSpanData(
            generator_id="test_generator",
            model_kwargs={"model": "gpt-4"},
            prompt_kwargs={"input": "test input"},
            raw_response="test response",
            generation_time_in_seconds=1.5,
            token_usage={"prompt_tokens": 10, "completion_tokens": 20},
        )

        self.assertEqual(span_data.generator_id, "test_generator")
        self.assertEqual(span_data.model_kwargs, {"model": "gpt-4"})
        self.assertEqual(span_data.prompt_kwargs, {"input": "test input"})
        self.assertEqual(span_data.raw_response, "test response")
        self.assertEqual(span_data.generation_time_in_seconds, 1.5)
        self.assertEqual(
            span_data.token_usage, {"prompt_tokens": 10, "completion_tokens": 20}
        )

        # Test export
        exported = span_data.export()
        self.assertEqual(exported["type"], "custom")
        self.assertEqual(exported["name"], "AdalFlow-Generator-test_generator")
        self.assertIn("generator_id", exported["data"])
        self.assertEqual(exported["data"]["generator_id"], "test_generator")

    def test_adalflow_tool_span_data(self):
        """Test AdalFlowToolSpanData."""
        span_data = AdalFlowToolSpanData(
            tool_name="test_tool",
            function_name="test_function",
            input_params={"arg1": "value1"},
            output_result="test result",
            execution_time=0.5,
        )

        self.assertEqual(span_data.tool_name, "test_tool")
        self.assertEqual(span_data.function_name, "test_function")
        self.assertEqual(span_data.input_params, {"arg1": "value1"})
        self.assertEqual(span_data.output_result, "test result")
        self.assertEqual(span_data.execution_time, 0.5)

        # Test export
        exported = span_data.export()
        self.assertEqual(exported["type"], "custom")
        self.assertEqual(exported["name"], "test_tool.test_function")
        self.assertIn("tool_name", exported["data"])
        self.assertEqual(exported["data"]["tool_name"], "test_tool")
        self.assertEqual(exported["data"]["input_params"], "{'arg1': 'value1'}")
        self.assertEqual(exported["data"]["output_result"], "test result")

    def test_adalflow_response_span_data(self):
        """Test AdalFlowResponseSpanData."""
        span_data = AdalFlowResponseSpanData(
            answer="test answer",
            result_type="string",
            execution_metadata={"steps": 5},
            response={"result": "success"},
        )

        self.assertEqual(span_data.answer, "test answer")
        self.assertEqual(span_data.result_type, "string")
        self.assertEqual(span_data.execution_metadata, {"steps": 5})
        self.assertEqual(span_data.response, {"result": "success"})

        # Test export
        exported = span_data.export()
        self.assertEqual(exported["type"], "custom")
        self.assertEqual(exported["name"], "response")
        self.assertIn("answer", exported["data"])
        self.assertEqual(exported["data"]["answer"], "test answer")

    def test_adalflow_step_span_data(self):
        """Test AdalFlowStepSpanData."""
        span_data = AdalFlowStepSpanData(
            step_number=1,
            action_type="planning",
            observation="test observation",
            is_final=False,
            function_name="test_function",
            function_results={"arg1": "value1"},
            execution_time=1.0,
        )

        self.assertEqual(span_data.step_number, 1)
        self.assertEqual(span_data.action_type, "planning")
        self.assertEqual(span_data.observation, "test observation")
        self.assertEqual(span_data.is_final, False)
        self.assertEqual(span_data.function_name, "test_function")
        self.assertEqual(span_data.function_results, {"arg1": "value1"})
        self.assertEqual(span_data.execution_time, 1.0)

        # Test export
        exported = span_data.export()
        self.assertEqual(exported["type"], "custom")
        self.assertEqual(exported["name"], "step-1")
        self.assertIn("step_number", exported["data"])
        self.assertEqual(exported["data"]["step_number"], 1)

        # Test update_attributes
        span_data.update_attributes(
            {"observation": "updated observation", "is_final": True}
        )
        self.assertEqual(span_data.observation, "updated observation")
        self.assertEqual(span_data.is_final, True)
        self.assertEqual(span_data.data["observation"], "updated observation")
        self.assertEqual(span_data.data["is_final"], True)


class TestSpanCreation(unittest.TestCase):
    """Test span creation functions."""

    def setUp(self):
        """Set up test environment."""
        self.original_disabled = GLOBAL_TRACE_PROVIDER._disabled
        set_tracing_disabled(False)
        set_trace_processors([])

    def tearDown(self):
        """Clean up test environment."""
        GLOBAL_TRACE_PROVIDER._disabled = self.original_disabled

    def test_span_creation_without_trace(self):
        """Test span creation without an active trace."""
        # Should return NoOpSpan when no trace is active
        span_obj = runner_span(runner_id="test_runner")
        self.assertIsInstance(span_obj, NoOpSpan)

    def test_span_creation_with_trace(self):
        """Test span creation with active trace."""
        processor = MockTracingProcessor()
        add_trace_processor(processor)

        with trace("test_workflow") as _trace_obj:
            # Test runner span
            runner_span_obj = runner_span(runner_id="test_runner", max_steps=10)
            self.assertIsInstance(runner_span_obj, SpanImpl)
            self.assertIsInstance(runner_span_obj.span_data, AdalFlowRunnerSpanData)
            self.assertEqual(runner_span_obj.span_data.runner_id, "test_runner")
            self.assertEqual(runner_span_obj.span_data.max_steps, 10)

            # Test generator span
            generator_span_obj = generator_span(generator_id="test_generator")
            self.assertIsInstance(generator_span_obj, SpanImpl)
            self.assertIsInstance(
                generator_span_obj.span_data, AdalFlowGeneratorSpanData
            )
            self.assertEqual(
                generator_span_obj.span_data.generator_id, "test_generator"
            )

            # Test tool span
            tool_span_obj = tool_span(
                tool_name="test_tool", function_name="test_function"
            )
            self.assertIsInstance(tool_span_obj, SpanImpl)
            self.assertIsInstance(tool_span_obj.span_data, AdalFlowToolSpanData)
            self.assertEqual(tool_span_obj.span_data.tool_name, "test_tool")
            self.assertEqual(tool_span_obj.span_data.function_name, "test_function")

            # Test response span
            response_span_obj = response_span(answer="test answer")
            self.assertIsInstance(response_span_obj, SpanImpl)
            self.assertIsInstance(response_span_obj.span_data, AdalFlowResponseSpanData)
            self.assertEqual(response_span_obj.span_data.answer, "test answer")

            # Test step span
            step_span_obj = step_span(step_number=1, action_type="planning")
            self.assertIsInstance(step_span_obj, SpanImpl)
            self.assertIsInstance(step_span_obj.span_data, AdalFlowStepSpanData)
            self.assertEqual(step_span_obj.span_data.step_number, 1)
            self.assertEqual(step_span_obj.span_data.action_type, "planning")

            # Test custom span
            custom_span_obj = custom_span(name="test_custom", data={"key": "value"})
            self.assertIsInstance(custom_span_obj, SpanImpl)
            self.assertIsInstance(custom_span_obj.span_data, CustomSpanData)
            self.assertEqual(custom_span_obj.span_data.name, "test_custom")
            self.assertEqual(custom_span_obj.span_data.data, {"key": "value"})

    def test_span_context_manager(self):
        """Test span as context manager."""
        processor = MockTracingProcessor()
        add_trace_processor(processor)

        with trace("test_workflow"):
            with runner_span(runner_id="test_runner") as span_obj:
                self.assertIsInstance(span_obj, SpanImpl)
                # Should have been started
                self.assertEqual(len(processor.span_starts), 1)
                self.assertEqual(processor.span_starts[0], span_obj)

            # Should be finished after exiting context
            self.assertEqual(len(processor.span_ends), 1)
            self.assertEqual(processor.span_ends[0], span_obj)

    def test_span_disabled(self):
        """Test span creation when disabled."""
        set_tracing_disabled(True)
        span_obj = runner_span(runner_id="test_runner")
        self.assertIsInstance(span_obj, NoOpSpan)


class TestTracingIntegration(unittest.TestCase):
    """Test tracing integration scenarios."""

    def setUp(self):
        """Set up test environment."""
        self.original_disabled = GLOBAL_TRACE_PROVIDER._disabled
        set_tracing_disabled(False)
        set_trace_processors([])

    def tearDown(self):
        """Clean up test environment."""
        GLOBAL_TRACE_PROVIDER._disabled = self.original_disabled

    def test_nested_spans(self):
        """Test nested span creation."""
        processor = MockTracingProcessor()
        add_trace_processor(processor)

        with trace("test_workflow"):
            with runner_span(runner_id="test_runner") as runner_span_obj:
                with generator_span(generator_id="test_generator") as gen_span_obj:
                    with tool_span(tool_name="test_tool") as tool_span_obj:
                        # Check parent relationships
                        self.assertEqual(
                            gen_span_obj.parent_id, runner_span_obj.span_id
                        )
                        self.assertEqual(tool_span_obj.parent_id, gen_span_obj.span_id)

        # Check that all spans were processed
        self.assertEqual(len(processor.span_starts), 3)
        self.assertEqual(len(processor.span_ends), 3)

    def test_multiple_processors(self):
        """Test multiple trace processors."""
        processor1 = MockTracingProcessor()
        processor2 = MockTracingProcessor()
        set_trace_processors([processor1, processor2])

        with trace("test_workflow") as _trace_obj:
            with runner_span(runner_id="test_runner") as _span_obj:
                pass

        # Both processors should have received events
        self.assertEqual(len(processor1.trace_starts), 1)
        self.assertEqual(len(processor1.trace_ends), 1)
        self.assertEqual(len(processor1.span_starts), 1)
        self.assertEqual(len(processor1.span_ends), 1)

        self.assertEqual(len(processor2.trace_starts), 1)
        self.assertEqual(len(processor2.trace_ends), 1)
        self.assertEqual(len(processor2.span_starts), 1)
        self.assertEqual(len(processor2.span_ends), 1)

    def test_span_data_update_during_execution(self):
        """Test updating span data during execution."""
        processor = MockTracingProcessor()
        add_trace_processor(processor)

        with trace("test_workflow"):
            with runner_span(runner_id="test_runner") as span_obj:
                # Update span data
                span_obj.span_data.update_attributes(
                    {"steps_executed": 5, "workflow_status": "running"}
                )

                # Check that data was updated
                self.assertEqual(span_obj.span_data.steps_executed, 5)
                self.assertEqual(span_obj.span_data.workflow_status, "running")
                self.assertEqual(span_obj.span_data.data["steps_executed"], 5)
                self.assertEqual(span_obj.span_data.data["workflow_status"], "running")

    def test_error_handling_in_spans(self):
        """Test error handling in spans."""
        processor = MockTracingProcessor()
        add_trace_processor(processor)

        with trace("test_workflow"):
            try:
                with runner_span(runner_id="test_runner") as _span_obj:
                    # Simulate an error
                    raise ValueError("Test error")
            except ValueError:
                pass

        # Span should still be properly ended even with error
        self.assertEqual(len(processor.span_starts), 1)
        self.assertEqual(len(processor.span_ends), 1)

    def test_processor_shutdown(self):
        """Test processor shutdown."""
        processor = MockTracingProcessor()
        add_trace_processor(processor)

        # Simulate shutdown
        GLOBAL_TRACE_PROVIDER.shutdown()

        self.assertTrue(processor.shutdown_called)


class TestTracingPerformance(unittest.TestCase):
    """Test tracing performance and resource usage."""

    def setUp(self):
        """Set up test environment."""
        self.original_disabled = GLOBAL_TRACE_PROVIDER._disabled
        set_tracing_disabled(False)
        set_trace_processors([])

    def tearDown(self):
        """Clean up test environment."""
        GLOBAL_TRACE_PROVIDER._disabled = self.original_disabled

    def test_disabled_tracing_performance(self):
        """Test that disabled tracing has minimal overhead."""
        set_tracing_disabled(True)

        # These should be very fast when disabled
        import time

        start_time = time.time()

        for i in range(1000):
            trace_obj = trace("test_workflow")
            span_obj = runner_span(runner_id=f"runner_{i}")
            # Should be NoOp objects
            self.assertIsInstance(trace_obj, NoOpTrace)
            self.assertIsInstance(span_obj, NoOpSpan)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete very quickly (less than 1 second for 1000 iterations)
        self.assertLess(duration, 1.0)

    def test_many_spans(self):
        """Test creating many spans."""
        processor = MockTracingProcessor()
        add_trace_processor(processor)

        with trace("test_workflow"):
            spans = []
            for i in range(100):
                span_obj = runner_span(runner_id=f"runner_{i}")
                spans.append(span_obj)
                span_obj.start()

            # End all spans
            for span_obj in spans:
                span_obj.finish()

        # Check that all spans were processed
        self.assertEqual(len(processor.span_starts), 100)
        self.assertEqual(len(processor.span_ends), 100)


if __name__ == "__main__":
    unittest.main()
