import pytest
from unittest.mock import Mock, patch, MagicMock
from adalflow.core.multirunner import MultiRunner
from adalflow.core.runner import Runner
from adalflow.core.agent import Agent
from adalflow.core.types import GeneratorOutput

class TestMultiRunner:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Mock Runner
        self.mock_runner = Mock(spec=Runner)
        self.mock_runner.call.return_value = GeneratorOutput(text="test response")
        self.mock_runner.acall.return_value = GeneratorOutput(text="test async response")
        
        # Mock Agent for runner creation
        self.mock_agent = Mock(spec=Agent)
        
        # Sample config
        self.sample_runner_config = {
            "agent": {
                "name": "test_agent",
                "tool_manager": {"tools": []},
                "system_prompt": "test",
                "model_client": {"component_name": "MockClient"},
                "model_kwargs": {"model": "test"}
            }
        }
        
    def test_add_get_runner(self):
        """Test adding and getting a runner"""
        mr = MultiRunner()
        mr.add_runner("test_runner", self.mock_runner)
        assert mr.get_runner("test_runner") == self.mock_runner

    def test_duplicate_runner(self):
        """Test adding duplicate runner name"""
        mr = MultiRunner()
        mr.add_runner("test_runner", self.mock_runner)
        with pytest.raises(ValueError):
            mr.add_runner("test_runner", self.mock_runner)

    def test_call(self):
        """Test calling a runner through multirunner"""
        mr = MultiRunner()
        mr.add_runner("test_runner", self.mock_runner)
        
        response = mr.call("test_runner", "test query")
        assert response.text == "test response"
        self.mock_runner.call.assert_called_once_with(
            user_query="test query",
            current_objective=None,
            memory=None,
            model_kwargs=None,
            use_cache=None,
            id=None
        )

    def test_update_runner(self):
        """Test updating a runner's configuration"""
        mr = MultiRunner()
        mr.add_runner("test_runner", self.mock_runner)
        
        update_config = {"agent_config": {"name": "updated"}}
        mr.update_runner("test_runner", **update_config)
        self.mock_runner.update_runner.assert_called_once_with(**update_config)

    def test_nonexistent_runner(self):
        """Test operations on non-existent runner"""
        mr = MultiRunner()
        with pytest.raises(KeyError):
            mr.get_runner("nonexistent")
        
        with pytest.raises(KeyError):
            mr.call("nonexistent", "test")
        
        with pytest.raises(KeyError):
            mr.update_runner("nonexistent", agent_config={})

    def test_create_runner_from_config(self):
        """Test creating a runner from config"""
        with patch('adalflow.core.runner.Runner.from_config') as mock_from_config:
            mock_from_config.return_value = self.mock_runner
            
            mr = MultiRunner()
            mr.create_runner_from_config("test_runner", self.sample_runner_config)
            
            assert mr.get_runner("test_runner") == self.mock_runner
            mock_from_config.assert_called_once_with(self.sample_runner_config)