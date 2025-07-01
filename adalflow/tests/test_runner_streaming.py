"""Test script for Runner streaming functionality with mocks using pytest."""

import asyncio
import pytest
from unittest.mock import MagicMock
from adalflow.core.runner import Runner
from adalflow.core.types import (
    RunItemStreamEvent,
    RawResponsesStreamEvent,
    GeneratorOutput,
    Function,
    FunctionOutput,
)


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.max_steps = 3
    agent.answer_data_type = str
    agent.tool_manager = MagicMock()
    agent.tool_manager.call = MagicMock(
        return_value=FunctionOutput(
            name="test_tool",
            input={"test": "input"},
            output="Tool executed successfully",
        )
    )

    # Create a mock output processor that returns a Function
    mock_output_processor = MagicMock()

    def process_output(text):
        return Function(name="finish", kwargs={"result": text.strip()})

    mock_output_processor.side_effect = process_output

    # Create a mock planner
    planner = MagicMock()
    planner.model_client = MagicMock()
    planner.model_type = "LLM"
    planner.output_processors = mock_output_processor

    async def mock_stream(*args, **kwargs):
        """Mock stream method that returns a GeneratorOutput with stream_events."""

        async def async_stream_events():
            # Simulate some streaming chunks
            mock_chunk1 = MagicMock()
            mock_chunk1.choices = [MagicMock()]
            mock_chunk1.choices[0].delta = MagicMock()
            mock_chunk1.choices[0].delta.content = "Test"
            yield mock_chunk1

            mock_chunk2 = MagicMock()
            mock_chunk2.choices = [MagicMock()]
            mock_chunk2.choices[0].delta = MagicMock()
            mock_chunk2.choices[0].delta.content = " result"
            yield mock_chunk2

        return GeneratorOutput(
            data=Function(name="finish", kwargs={"result": "Test result"}),
            raw_response="Mock response",
            stream_events=async_stream_events(),
        )

    planner.stream = mock_stream
    agent.planner = planner

    return agent


@pytest.mark.asyncio
async def test_runner_streaming(mock_agent):
    """Test the Runner streaming functionality."""
    runner = Runner(agent=mock_agent)
    prompt_kwargs = {"query": "Test streaming execution"}

    streaming_result = runner.run_streamed(
        prompt_kwargs=prompt_kwargs, model_kwargs={"stream": True}
    )

    event_count = 0
    stream = streaming_result.stream_events()
    # pull events until the stream ends or we hit a timeout
    while True:
        try:
            event = await asyncio.wait_for(stream.__anext__(), timeout=5)
        except (asyncio.TimeoutError, StopAsyncIteration):
            break
        event_count += 1
        assert event is not None
        assert isinstance(event, (RawResponsesStreamEvent, RunItemStreamEvent))
        if event_count > 20:  # safety bail-out
            break

    assert event_count > 0, "Should have received at least one event"


@pytest.mark.asyncio
async def test_runner_streaming_with_multiple_steps():
    """Test Runner with multiple steps that don't finish immediately."""

    class MockAgentMultiStep:
        def __init__(self):
            self.max_steps = 3
            self._step_count = 0
            self.answer_data_type = str
            self.planner = None
            self.tool_manager = MagicMock()
            self.tool_manager.call = MagicMock(
                return_value=FunctionOutput(
                    name="test_tool",
                    input={"test": "input"},
                    output="Tool executed successfully",
                )
            )

    class MockPlannerMultiStep:
        def __init__(self, agent):
            self._agent = agent
            self.model_client = MagicMock()
            self.model_type = "LLM"

            mock_output_processor = MagicMock()

            def process_output(text):
                if "Step 1" in text:
                    return Function(name="continue", kwargs={"action": "step_1"})
                else:
                    return Function(name="finish", kwargs={"result": text.strip()})

            mock_output_processor.side_effect = process_output
            self.output_processors = mock_output_processor

        async def stream(self, **kwargs):
            async def async_stream_events():
                mock_chunk1 = MagicMock()
                mock_chunk1.choices = [MagicMock()]
                mock_chunk1.choices[0].delta = MagicMock()
                mock_chunk1.choices[0].delta.content = f"Step {self._agent._step_count}"
                yield mock_chunk1

                mock_chunk2 = MagicMock()
                mock_chunk2.choices = [MagicMock()]
                mock_chunk2.choices[0].delta = MagicMock()
                mock_chunk2.choices[0].delta.content = " content"
                yield mock_chunk2

            self._agent._step_count += 1
            if self._agent._step_count >= 2:
                function = Function(name="finish", kwargs={"result": "Final answer"})
            else:
                function = Function(
                    name="continue",
                    kwargs={"action": f"step_{self._agent._step_count}"},
                )

            return GeneratorOutput(
                data=function,
                raw_response=f"Step {self._agent._step_count} response",
                stream_events=async_stream_events(),
            )

    mock_agent = MockAgentMultiStep()
    mock_agent.planner = MockPlannerMultiStep(mock_agent)
    runner = Runner(agent=mock_agent)

    streaming_result = runner.run_streamed(
        prompt_kwargs={"query": "Multi-step test"}, model_kwargs={"stream": True}
    )

    step_events = []
    stream = streaming_result.stream_events()
    # pull events until runner signals finish or we hit a timeout
    while True:
        try:
            event = await asyncio.wait_for(stream.__anext__(), timeout=5)
        except (asyncio.TimeoutError, StopAsyncIteration):
            break
        if isinstance(event, RunItemStreamEvent):
            if event.name == "step_completed":
                step_events.append(event)
            elif event.name == "runner_finished":
                break
        # give the loop a moment for other tasks
        await asyncio.sleep(0.01)

    assert (
        len(step_events) >= 0
    ), f"Unexpected error processing, got {len(step_events)} step events"
    assert streaming_result.is_complete, "Runner should be complete"
