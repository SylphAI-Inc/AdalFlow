import pytest
from unittest.mock import Mock, patch, MagicMock
from adalflow.core.agent import Agent
from adalflow.core.tool_manager import ToolManager
from adalflow.core.types import GeneratorOutput
from adalflow.core.model_client import ModelClient

class TestAgent:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Mock dependencies
        self.mock_tool_manager = Mock(spec=ToolManager)
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
        
        # Mock Generator
        self.mock_generator = Mock()
        self.mock_generator.return_value = GeneratorOutput(text="test response")
        self.mock_generator.acall.return_value = GeneratorOutput(text="test async response")
        
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

    def test_agent_call(self):
        """Test agent call method"""
        agent = Agent(
            name="test",
            tool_manager=self.mock_tool_manager,
            system_prompt="test",
            model_client=self.mock_model_client,
            model_kwargs={}
        )
        
        response = agent.call("test query")
        assert response.text == "test response"
        self.mock_generator.assert_called_once()

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
        agent.update_agent(
            name="updated",
            tool_manager=new_tm,
            system_prompt="updated prompt"
        )
        
        assert agent.name == "updated"
        assert agent.tool_manager == new_tm
        assert agent.system_prompt == "updated prompt"