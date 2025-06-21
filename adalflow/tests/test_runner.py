import pytest
from unittest.mock import Mock, patch, MagicMock
from adalflow.core.runner import Runner
from adalflow.core.agent import Agent
from adalflow.core.types import GeneratorOutput

class TestRunner:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Mock Agent
        self.mock_agent = Mock(spec=Agent)
        self.mock_agent.call.return_value = GeneratorOutput(text="test response")
        self.mock_agent.acall.return_value = GeneratorOutput(text="test async response")
        
        # Sample config
        self.sample_config = {
            "agent": {
                "name": "test_agent",
                "tool_manager": {"tools": []},
                "system_prompt": "test",
                "model_client": {"component_name": "MockClient"},
                "model_kwargs": {"model": "test"}
            }
        }
        
    def test_runner_init(self):
        """Test runner initialization"""
        runner = Runner(agent=self.mock_agent)
        assert runner.agent == self.mock_agent

    def test_runner_call(self):
        """Test runner call method"""
        runner = Runner(agent=self.mock_agent)
        response = runner.call("test query")
        assert response.text == "test response"
        self.mock_agent.call.assert_called_once()

    def test_runner_acall(self):
        """Test runner async call method"""
        runner = Runner(agent=self.mock_agent)
        response = runner.acall("test query")
        assert response.text == "test async response"
        self.mock_agent.acall.assert_called_once()

    def test_runner_update_runner(self):
        """Test updating runner configuration"""
        runner = Runner(agent=self.mock_agent)
        new_agent = Mock(spec=Agent)
        
        # Test updating with agent instance
        runner.update_runner(agent=new_agent)
        assert runner.agent == new_agent
        
        # Test updating with agent config
        with patch('adalflow.core.agent.Agent.from_config') as mock_from_config:
            mock_agent = Mock(spec=Agent)
            mock_from_config.return_value = mock_agent
            
            runner.update_runner(agent_config={"name": "test"})
            mock_from_config.assert_called_once_with({"name": "test"})
            assert runner.agent == mock_agent

    def test_runner_from_config(self):
        """Test creating runner from config"""
        with patch('adalflow.core.agent.Agent.from_config') as mock_from_config:
            mock_agent = Mock(spec=Agent)
            mock_from_config.return_value = mock_agent
            
            runner = Runner.from_config(self.sample_config)
            assert runner.agent == mock_agent
            mock_from_config.assert_called_once_with(self.sample_config["agent"])