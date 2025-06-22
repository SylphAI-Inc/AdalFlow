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
        self.mock_runner.call.return_value = GeneratorOutput(data="test response")
        self.mock_runner.acall.return_value = GeneratorOutput(data="test async response")
        
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
        
    def create_multi_runner(self):
        """Helper to create a MultiRunner with an empty runners dict"""
        return MultiRunner(runners={})
        
    def test_add_get_runner(self):
        """Test adding and getting a runner"""
        mr = self.create_multi_runner()
        mr.add_runner("test_runner", self.mock_runner)
        assert mr.get_runner("test_runner") == self.mock_runner

    def test_duplicate_runner(self):
        """Test adding duplicate runner name"""
        mr = self.create_multi_runner()
        mr.add_runner("test_runner", self.mock_runner)
        with pytest.raises(ValueError):
            mr.add_runner("test_runner", self.mock_runner)

    def test_call(self):
        """Test calling a runner through multirunner"""
        mr = self.create_multi_runner()
        mr.add_runner("test_runner", self.mock_runner)
        
        response = mr.call("test_runner", "test query")
        assert response.data == "test response"
        self.mock_runner.call.assert_called_once_with(
            user_query="test query",
            current_objective=None,
            memory=None,
            model_kwargs={},  # Changed from None to {}
            use_cache=None,
            id=None,
            context=None  # Added context parameter
        )

    def test_update_runner(self):
        """Test updating a runner's configuration"""
        mr = self.create_multi_runner()
        mr.add_runner("test_runner", self.mock_runner)
        
        update_config = {"agent_config": {"name": "updated"}}
        mr.update_runner("test_runner", **update_config)
        self.mock_runner.update_runner.assert_called_once_with(**update_config)

    def test_nonexistent_runner(self):
        """Test operations on non-existent runner"""
        mr = self.create_multi_runner()
        with pytest.raises(KeyError):
            mr.get_runner("nonexistent")
        
        with pytest.raises(KeyError):
            mr.call("nonexistent", "test")
        
        with pytest.raises(KeyError):
            mr.update_runner("nonexistent", agent_config={})

    def test_add_runner(self):
        """Test adding a runner"""
        mr = self.create_multi_runner()
        mr.add_runner("test_runner", self.mock_runner)
        assert mr.get_runner("test_runner") == self.mock_runner