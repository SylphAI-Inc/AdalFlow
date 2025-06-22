import pytest
from unittest.mock import Mock, patch, MagicMock
from adalflow.core.agent import Agent
from adalflow.core.tool_manager import ToolManager
from adalflow.core.types import GeneratorOutput
from adalflow.core.model_client import ModelClient
import asyncio

class TestAgent:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Mock dependencies
        self.mock_tool_manager = Mock(spec=ToolManager)
        self.mock_tool_manager.get_all_tools = Mock(return_value=[])
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
        
        # Create a mock generator instance with data attribute
        self.mock_generator_instance = Mock()
        self.mock_generator_instance.data = "test response"
        
        # Create a mock for the generator class
        self.mock_generator = Mock(return_value=self.mock_generator_instance)
        self.mock_generator.acall = Mock(return_value=self.mock_generator_instance)
        
        # Patch Generator class
        self.generator_patcher = patch('adalflow.core.agent.Generator', return_value=self.mock_generator)
        self.mock_generator_class = self.generator_patcher.start()
        
        yield
        
        # Cleanup
        self.generator_patcher.stop()

    def test_agent_init_with_config_generator(self):
        """Test agent initialization with config_generator"""
        config_generator = {
            "model_client": {"component_name": "MockClient"},
            "model_kwargs": {"model": "test"}
        }
        agent = Agent(
            name="test",
            tool_manager=self.mock_tool_manager,
            system_prompt="test",
            config_generator=config_generator
        )
        assert agent.config_generator == config_generator

    def test_agent_init_with_model_client(self):
        """Test agent initialization with direct model client"""
        agent = Agent(
            name="test",
            tool_manager=self.mock_tool_manager,
            system_prompt="test",
            model_client=self.mock_model_client,
            model_kwargs={"model": "test"}
        )
        assert agent.name == "test"
        assert agent.system_prompt == "test"

    def test_agent_init_missing_required_args(self):
        """Test agent initialization with missing required arguments"""
        with pytest.raises(TypeError):
            Agent()  # Missing required arguments

    def test_agent_call(self):
        """Test agent call method"""
        agent = Agent(
            name="test",
            tool_manager=self.mock_tool_manager,
            system_prompt="test",
            model_client=self.mock_model_client,
            model_kwargs={}
        )
        
        # Create a mock response with a data attribute
        mock_response = Mock()
        mock_response.data = "test response"
        
        # Configure the generator instance to return our mock response
        agent.generator.return_value = mock_response
    
        response = agent.call("test query")
        assert response.data == "test response"
        agent.generator.assert_called_once()

    def test_agent_call_with_all_parameters(self):
        """Test agent call with all optional parameters"""
        agent = Agent(
            name="test",
            tool_manager=self.mock_tool_manager,
            system_prompt="test",
            model_client=self.mock_model_client,
            model_kwargs={}
        )
        
        response = agent.call(
            user_query="test query",
            current_objective="test objective",
            memory="test memory",
            model_kwargs={"temperature": 0.7},
            use_cache=True,
            id="test_id",
            context=["context1", "context2"]
        )
        assert response.data == "test response"
        self.mock_generator.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_acall(self):
        """Test async agent call method"""
        agent = Agent(
            name="test",
            tool_manager=self.mock_tool_manager,
            system_prompt="test",
            model_client=self.mock_model_client,
            model_kwargs={}
        )
        
        response = await agent.acall("test query")
        assert response.data == "test response"
        self.mock_generator.acall.assert_called_once()

    def test_agent_from_config(self):
        """Test creating agent from config"""
        with patch('adalflow.core.agent.ToolManager') as mock_tm_class:
            mock_tm_instance = Mock()
            mock_tm_class.from_config.return_value = mock_tm_instance
            
            agent = Agent.from_config(self.sample_config)
            
            assert agent.name == "test_agent"
            assert agent.system_prompt == "Test system prompt"
            mock_tm_class.from_config.assert_called_once_with(self.sample_config["tool_manager"])

    def test_agent_update_agent(self):
        """Test updating agent configuration"""
        agent = Agent(
            name="test",
            tool_manager=self.mock_tool_manager,
            system_prompt="test",
            model_client=self.mock_model_client,
            model_kwargs={}
        )
        
        new_tm = Mock(spec=ToolManager)
        new_config = {"model_client": {"component_name": "NewClient"}}
        
        agent.update_agent(
            name="updated",
            tool_manager=new_tm,
            system_prompt="updated prompt",
            generator_config=new_config,
            current_mode="test_mode"
        )
        
        assert agent.name == "updated"
        assert agent.tool_manager == new_tm
        assert agent.system_prompt == "updated prompt"
        assert agent.current_mode == "test_mode"
        assert agent.config_generator == new_config

    def test_get_prompt(self):
        """Test getting formatted prompt"""
        agent = Agent(
            name="test",
            tool_manager=self.mock_tool_manager,
            system_prompt="test {name}",
            model_client=self.mock_model_client,
            model_kwargs={}
        )
        
        with patch.object(agent.generator, 'get_prompt', return_value="formatted prompt") as mock_get_prompt:
            result = agent.get_prompt(name="test_name")
            assert result == "formatted prompt"
            mock_get_prompt.assert_called_once()

    def test_get_all_tools(self):
        """Test getting all tools from tool manager"""
        expected_tools = [{"name": "test_tool", "description": "A test tool"}]
        self.mock_tool_manager.get_all_tools.return_value = expected_tools
        
        agent = Agent(
            name="test",
            tool_manager=self.mock_tool_manager,
            system_prompt="test",
            model_client=self.mock_model_client,
            model_kwargs={}
        )
        
        tools = agent.get_all_tools()
        assert tools == expected_tools
        self.mock_tool_manager.get_all_tools.assert_called_once()

    def test_create_prompt_kwargs(self):
        """Test _create_prompt_kwargs method"""
        agent = Agent(
            name="test",
            tool_manager=self.mock_tool_manager,
            system_prompt="Test {current_mode}",
            model_client=self.mock_model_client,
            model_kwargs={}
        )
        
        agent.current_mode = "test_mode"
        prompt_kwargs = agent._create_prompt_kwargs(
            user_query="test query",
            current_objective="test objective",
            memory="test memory",
            context=["context1", "context2"]
        )
        
        assert prompt_kwargs == {
            "task_desc_str": "Test test_mode",
            "input_str": "test query",
            "current_objective": "test objective",
            "chat_history_str": "test memory",
            "context_str": ["context1", "context2"]
        }

    def test_agent_serialization_deserialization(self):
        """Test serializing agent to dict and recreating from config"""
        # Create initial agent
        agent = Agent(
            name="test_agent",
            tool_manager=self.mock_tool_manager,
            system_prompt="Test system prompt",
            model_client=self.mock_model_client,
            model_kwargs={"temperature": 0.7, "max_tokens": 100},
            current_mode="test_mode"
        )

        # Convert agent to state dict
        state_dict = agent.return_state_dict()

        # Mock ToolManager.from_config to return our mock tool manager
        with patch('adalflow.core.agent.ToolManager.from_config', return_value=self.mock_tool_manager):
            # Create a new agent from the state dict
            new_agent = Agent.from_config(state_dict)
        
        # Verify all attributes are preserved
        assert new_agent.name == agent.name
        assert new_agent.system_prompt == agent.system_prompt
        assert new_agent.current_mode == agent.current_mode
        assert new_agent.config_generator == agent.config_generator
        # Verify tool manager was recreated
        assert isinstance(new_agent.tool_manager, Mock)

    @pytest.mark.asyncio
    async def test_agent_acall(self):
        """Test async agent call method"""
        agent = Agent(
            name="test",
            tool_manager=self.mock_tool_manager,
            system_prompt="test",
            model_client=self.mock_model_client,
            model_kwargs={}
        )
        
        # Create an awaitable mock
        mock_async_result = asyncio.Future()
        mock_async_result.set_result(GeneratorOutput(data="test response"))
        agent.generator.acall.return_value = mock_async_result
        
        response = await agent.acall("test query")
        assert response.data == "test response"
        agent.generator.acall.assert_called_once()

    def test_agent_call_with_all_parameters(self):
        """Test agent call with all optional parameters"""
        agent = Agent(
            name="test",
            tool_manager=self.mock_tool_manager,
            system_prompt="test",
            model_client=self.mock_model_client,
            model_kwargs={}
        )
        
        # Configure the generator instance's return value
        mock_response = Mock()
        mock_response.data = "test response"
        agent.generator.return_value = mock_response
        
        response = agent.call(
            user_query="test query",
            current_objective="test objective",
            memory="test memory",
            model_kwargs={"temperature": 0.7},
            use_cache=True,
            id="test_id",
            context=["context1", "context2"]
        )
        assert response.data == "test response"
        agent.generator.assert_called_once()

    def test_create_prompt_kwargs(self):
        """Test _create_prompt_kwargs method"""
        agent = Agent(
            name="test",
            tool_manager=self.mock_tool_manager,
            system_prompt="Test {current_mode}",
            model_client=self.mock_model_client,
            model_kwargs={}
        )
        
        agent.current_mode = "test_mode"
        prompt_kwargs = agent._create_prompt_kwargs(
            user_query="test query",
            current_objective="test objective",
            memory="test memory",
            context=["context1", "context2"]
        )
        
        # The task_desc_str should not be formatted yet as it's done in the Prompt class
        assert prompt_kwargs == {
            "task_desc_str": "Test {current_mode}",
            "input_str": "test query",
            "current_objective": "test objective",
            "chat_history_str": "test memory",
            "context_str": ["context1", "context2"]
        }