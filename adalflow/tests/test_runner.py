import pytest
import json
from unittest.mock import Mock, patch, AsyncMock, call, ANY
from adalflow.core.runner import Runner
from adalflow.core.agent import Agent
from adalflow.core.types import (
    GeneratorOutput, 
    StepOutput, 
    Function, 
    FunctionOutput,
    FunctionExpression
)
from adalflow.core.exceptions import ToolExecutionError

class TestRunner:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Create test data
        self.test_function = Function(
            name="test_tool",
            arguments=json.dumps({"param1": "value1"})
        )
        
        self.test_step_output = StepOutput(
            step=1,
            action="test_action",
            function=self.test_function,
            observation={"result": "test result"}
        )
        
        self.generator_output = GeneratorOutput(
            data=self.test_step_output,
            raw_response="raw response"
        )
        
        # Mock Agent
        self.mock_agent = Mock()
        self.mock_agent.call = Mock(return_value=self.generator_output)
        self.mock_agent.acall = AsyncMock(return_value=self.generator_output)
        self.mock_agent.tool_manager.get_tool.return_value = Mock()
        self.mock_agent.tool_manager.execute_func.return_value = "tool result"
        self.mock_agent.tool_manager.aexecute_func = AsyncMock(return_value="async tool result")
        self.mock_agent.is_training.return_value = False
        
        # Mock output type
        self.mock_output_type = Mock()
        self.mock_output_type.__name__ = "MockOutput"
        
        # Create runner
        self.runner = Runner(
            agent=self.mock_agent,
            output_type=self.mock_output_type,
            max_steps=5
        )

    # Test Initialization
    def test_init(self):
        """Test runner initialization"""
        assert self.runner.agent == self.mock_agent
        assert self.runner.max_steps == 5
        assert len(self.runner.step_history) == 0
        assert self.runner.config.output_type == self.mock_output_type

    # Test Call Methods
    def test_call(self):
        """Test synchronous call"""
        prompt_kwargs = {"input": "test input"}
        model_kwargs = {"temperature": 0.7}
        
        step_history, result = self.runner.call(
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
            use_cache=True,
            id="test_call"
        )
        
        # Verify agent was called correctly
        self.mock_agent.call.assert_called_once_with(
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
            use_cache=True,
            id="test_call"
        )
        
        # Verify step history
        assert len(step_history) == 1
        assert step_history[0] == self.generator_output
        
        # Verify result was processed
        assert result is not None

    @pytest.mark.asyncio
    async def test_acall(self):
        """Test asynchronous call"""
        prompt_kwargs = {"input": "async test"}
        
        step_history, result = await self.runner.acall(
            prompt_kwargs=prompt_kwargs,
            model_kwargs=None,
            use_cache=False,
            id="test_acall"
        )
        
        # Verify agent was called correctly
        self.mock_agent.acall.assert_awaited_once_with(
            prompt_kwargs=prompt_kwargs,
            model_kwargs=None,
            use_cache=False,
            id="test_acall"
        )
        
        # Verify step history
        assert len(step_history) == 1
        assert step_history[0] == self.generator_output
        
        # Verify result was processed
        assert result is not None

    # Test Function Call Processing
    def test_process_function_calls(self):
        """Test function call processing"""
        function = Function(
            name="test_tool",
            arguments=json.dumps({"param1": "value1"})
        )
        
        result = self.runner._process_function_calls(function)
        
        # Verify function was executed
        self.mock_agent.tool_manager.execute_func.assert_called_once_with(function)
        assert isinstance(result, FunctionOutput)
        assert result.name == "test_tool"
        assert result.output == "tool result"

    @pytest.mark.asyncio
    async def test_aprocess_function_calls(self):
        """Test async function call processing"""
        function = Function(
            name="test_tool",
            arguments=json.dumps({"param1": "value1"})
        )
        
        result = await self.runner._aprocess_function_calls(function)
        
        # Verify function was executed
        self.mock_agent.tool_manager.aexecute_func.assert_awaited_once_with(function)
        assert isinstance(result, FunctionOutput)
        assert result.name == "test_tool"
        assert result.output == "async tool result"

    # Test Step Processing
    def test_process_data(self):
        """Test data processing"""
        step_output = StepOutput(
            step=1,
            action="test",
            function=None,
            observation={"field1": "value1", "field2": 42}
        )
        
        # Configure mock to return the observation dict as is
        self.mock_output_type.return_value = step_output.observation
        
        result = self.runner._process_data(step_output, "test_id")
        
        assert result == step_output.observation
        self.mock_output_type.assert_called_once_with(**step_output.observation)

    # Test Edge Cases
    def test_max_steps(self):
        """Test max steps handling"""
        # Create a runner with max_steps=1
        runner = Runner(
            agent=self.mock_agent,
            output_type=self.mock_output_type,
            max_steps=1
        )
        
        # First call should work
        step_history, result = runner.call(prompt_kwargs={"input": "test"})
        assert len(step_history) == 1
        
        # Second call should not exceed max_steps
        step_history, result = runner.call(prompt_kwargs={"input": "test2"})
        assert len(step_history) == 1  # Should not have increased

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in async call"""
        self.mock_agent.acall.side_effect = Exception("Test error")
        
        with pytest.raises(Exception, match="Test error"):
            await self.runner.acall(prompt_kwargs={"input": "test"})

    # Test Training Mode
    def test_training_mode(self):
        """Test behavior in training mode"""
        # Configure agent to be in training mode
        self.mock_agent.is_training.return_value = True
        
        # Create a training step output
        training_step = StepOutput(
            step=1,
            action="train",
            function=None,
            observation={"loss": 0.5, "accuracy": 0.9}
        )
        
        training_output = GeneratorOutput(
            data=GeneratorOutput(data=training_step),
            raw_response="training response"
        )
        
        self.mock_agent.call.return_value = training_output
        
        step_history, result = self.runner.call(prompt_kwargs={"input": "train"})
        
        assert len(step_history) == 1
        assert step_history[0] == training_output
        assert result == training_step.observation

    # Test Function Chain
    def test_function_chaining(self):
        """Test chaining multiple function calls"""
        # First function call
        func1 = Function(name="first_func", arguments=json.dumps({"param": "value"}))
        step1 = StepOutput(
            step=1,
            action="function",
            function=func1,
            observation=None
        )
        
        # Second function call with result from first
        func2 = Function(name="second_func", arguments=json.dumps({"result": "tool result"}))
        step2 = StepOutput(
            step=2,
            action="function",
            function=func2,
            observation={"final": "result"}
        )
        
        # Configure mock to return different outputs
        self.mock_agent.call.side_effect = [
            GeneratorOutput(data=step1),
            GeneratorOutput(data=step2)
        ]
        
        # First call should process func1
        step_history, result = self.runner.call(prompt_kwargs={"input": "chain"})
        
        # Should have called execute_func with func1
        self.mock_agent.tool_manager.execute_func.assert_called_once_with(func1)
        
        # Second call should process func2 with result from func1
        step_history, result = self.runner.call(
            prompt_kwargs={"function_results": "tool result"}
        )
        
        # Should have called execute_func with func2
        assert self.mock_agent.tool_manager.execute_func.call_count == 2
        assert "final" in result