import pytest
import json
from unittest.mock import Mock, patch, AsyncMock, call, ANY
from adalflow.core.agent import Agent
from adalflow.core.tool_manager import ToolManager
from adalflow.core.types import GeneratorOutput, StepOutput, Function, FunctionOutput
from adalflow.core.model_client import ModelClient
from adalflow.core.exceptions import ToolExecutionError

class TestAgent:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Mock dependencies
        self.mock_tool_manager = Mock(spec=ToolManager)
        self.mock_tool_manager.get_all_tools = Mock(return_value=[])
        self.mock_tool_manager.get_tool = Mock(return_value=Mock())
        self.mock_tool_manager.execute_func = Mock(return_value="tool result")
        self.mock_tool_manager.aexecute_func = AsyncMock(return_value="async tool result")
        
        self.mock_model_client = Mock(spec=ModelClient)
        self.mock_model_client.to_dict.return_value = {"component_name": "MockClient"}
        
        # Sample config
        self.sample_config = {
            "name": "test_agent",
            "tool_manager": {"tools": []},
            "system_prompt": "Test system prompt",
            "model_client": {"component_name": "MockClient"},
            "model_kwargs": {"model": "test"}
        }
        
        # Create test functions and outputs
        self.test_function = Function(
            name="test_func",
            arguments=json.dumps({"arg1": "value1"})
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
        
        # Mock generator
        self.mock_generator = Mock()
        self.mock_generator.call = Mock(return_value=self.generator_output)
        self.mock_generator.acall = AsyncMock(return_value=self.generator_output)
        
        # Patch Generator class
        self.generator_patcher = patch('adalflow.core.agent.Generator', return_value=self.mock_generator)
        self.mock_generator_class = self.generator_patcher.start()
        
        yield
        
        # Cleanup
        self.generator_patcher.stop()

    # Test Initialization
    def test_init_with_tools(self):
        """Test agent initialization with tools"""
        tools = [Mock(), Mock()]
        agent = Agent(
            name="test_agent",
            system_prompt="Test",
            model_client=self.mock_model_client,
            tools=tools
        )
        
        assert agent.name == "test_agent"
        assert len(agent.tool_manager.tools) == 2

    # Test Call Methods
    def test_call(self):
        """Test agent call method"""
        agent = Agent(
            name="test_agent",
            system_prompt="Test",
            model_client=self.mock_model_client
        )
        
        prompt_kwargs = {"input": "test input", "context": "test context"}
        model_kwargs = {"temperature": 0.7}
        
        result = agent.call(
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
            use_cache=True,
            id="test_id"
        )
        
        # Verify generator was called with correct arguments
        self.mock_generator.call.assert_called_once_with(
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
            use_cache=True,
            id="test_id"
        )
        
        assert result == self.generator_output

    @pytest.mark.asyncio
    async def test_acall(self):
        """Test agent async call method"""
        agent = Agent(
            name="test_agent",
            system_prompt="Test",
            model_client=self.mock_model_client
        )
        
        prompt_kwargs = {"input": "test input"}
        
        result = await agent.acall(
            prompt_kwargs=prompt_kwargs,
            model_kwargs=None,
            use_cache=False,
            id="async_test"
        )
        
        # Verify generator was called with correct arguments
        self.mock_generator.acall.assert_awaited_once_with(
            prompt_kwargs=prompt_kwargs,
            model_kwargs=None,
            use_cache=False,
            id="async_test"
        )
        
        assert result == self.generator_output

    # Test Tool Execution
    def test_execute_tool(self):
        """Test tool execution"""
        agent = Agent(
            name="test_agent",
            system_prompt="Test",
            model_client=self.mock_model_client
        )
        
        result = agent.tool_manager.execute_func(self.test_function)
        assert result == "tool result"
        self.mock_tool_manager.execute_func.assert_called_once_with(self.test_function)

    @pytest.mark.asyncio
    async def test_aexecute_tool(self):
        """Test async tool execution"""
        agent = Agent(
            name="test_agent",
            system_prompt="Test",
            model_client=self.mock_model_client
        )
        
        result = await agent.tool_manager.aexecute_func(self.test_function)
        assert result == "async tool result"
        self.mock_tool_manager.aexecute_func.assert_awaited_once_with(self.test_function)

    # Test Error Cases
    def test_call_with_invalid_prompt_kwargs(self):
        """Test call with invalid prompt kwargs"""
        agent = Agent(
            name="test_agent",
            system_prompt="Test",
            model_client=self.mock_model_client
        )
        
        with pytest.raises(ValueError):
            agent.call(prompt_kwargs=None)

    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        """Test tool execution error handling"""
        self.mock_tool_manager.execute_func.side_effect = ToolExecutionError("Tool failed")
        
        agent = Agent(
            name="test_agent",
            system_prompt="Test",
            model_client=self.mock_model_client
        )
        
        with pytest.raises(ToolExecutionError):
            agent.tool_manager.execute_func(self.test_function)

    # Test Configuration
    def test_update_config(self):
        """Test updating agent configuration"""
        agent = Agent(
            name="test_agent",
            system_prompt="Test",
            model_client=self.mock_model_client
        )
        
        new_config = {
            "name": "updated_agent",
            "system_prompt": "Updated prompt",
            "model_kwargs": {"temperature": 0.8}
        }
        
        agent.update_agent_config(**new_config)
        
        assert agent.name == "updated_agent"
        assert agent.system_prompt == "Updated prompt"
        assert agent.model_kwargs["temperature"] == 0.8

    # Test Serialization
    def test_to_dict(self):
        """Test agent serialization to dictionary"""
        agent = Agent(
            name="test_agent",
            system_prompt="Test",
            model_client=self.mock_model_client,
            tools=[Mock()]
        )
        
        agent_dict = agent.to_dict()
        
        assert agent_dict["name"] == "test_agent"
        assert agent_dict["system_prompt"] == "Test"
        assert "tools" in agent_dict
        assert "model_client" in agent_dict

    @pytest.mark.asyncio
    async def test_concurrent_calls(self):
        """Test multiple concurrent async calls"""
        agent = Agent(
            name="test_agent",
            system_prompt="Test",
            model_client=self.mock_model_client
        )
        
        # Create multiple tasks
        tasks = [
            agent.acall(prompt_kwargs={"input": f"test {i}"}, id=f"task-{i}")
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(isinstance(r, GeneratorOutput) for r in results)
        assert self.mock_generator.acall.await_count == 5