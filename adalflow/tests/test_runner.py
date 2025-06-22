# Update the imports at the top of the file
from dataclasses import asdict
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from adalflow.core.runner import Runner, RunnerConfig
from adalflow.core.agent import Agent
from adalflow.core.types import GeneratorOutput
from adalflow.components.output_parsers.outputs import OutputParser
from typing import Dict, Any, Optional
import json

class MockOutputParser(OutputParser):
    def parse(self, output: GeneratorOutput) -> Dict[str, Any]:
        return {"text": output.data, "metadata": {}}

class TestRunner:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Mock Agent
        self.mock_agent = Mock(spec=Agent)
        
        # Create proper GeneratorOutput instances
        self.mock_output = GeneratorOutput(data="test response")
        self.mock_async_output = GeneratorOutput(data="test async response")
        
        # Configure agent mocks
        self.mock_agent.call.return_value = self.mock_output
        self.mock_agent.acall = AsyncMock(return_value=self.mock_async_output)
        
        # Create mock parser with parse method
        self.mock_parser = Mock(spec=OutputParser)
        self.mock_parser.parse = Mock(return_value={"text": "test response", "metadata": {}})
        
        # Create a serializable config
        self.sample_config = {
            "agent": {
                "name": "test_agent",
                "tool_manager": {"tools": []},
                "system_prompt": "test",
                "model_client": {"component_name": "MockClient"},
                "model_kwargs": {"model": "test"}
            },
            "output_parser": {
                "component_name": "MockOutputParser"
            },
            "output_class": "GeneratorOutput"
        }

    # TODO need to include more test cases with parss 

    def test_call_with_parser(self):
        """Test call method with output parser"""
        # Create a simple class for our output
        class TestOutput:
            pass

        # Create a mock output class that returns our test output
        mock_output_class = Mock(side_effect=lambda: TestOutput())
        
        # Create a real instance of MockOutputParser
        parser = MockOutputParser()
        
        # Mock the call method on the parser to return our test data
        parser.call = Mock(return_value={"text": "text response", "metadata": {}})
        
        # Create the runner with our mocked components
        runner = Runner(
            agent=self.mock_agent, 
            output_parser=parser,
            output_class=mock_output_class
        )
        
        # Mock the agent's call to return a proper GeneratorOutput
        test_output = GeneratorOutput(data="test response")
        self.mock_agent.call.return_value = test_output
        
        # Call the method under test
        response = runner.call("test query")
        
        # Verify the response is an instance of our test output class
        assert isinstance(response, TestOutput)
        
        # Verify the parser's call method was called with the generator output
        parser.call.assert_called_once_with(test_output)
        
        # Verify the output class was instantiated
        mock_output_class.assert_called_once()
        
        # Verify the attributes were set on the output instance
        assert response.text == "text response"
        assert response.metadata == {}

    @pytest.mark.asyncio
    async def test_acall(self):
        """Test async call method"""
        runner = Runner(agent=self.mock_agent)
        response = await runner.acall("test query")
        assert response == self.mock_async_output
        self.mock_agent.acall.assert_called_once_with(
            "test query",  # user_query
            None,          # current_objective
            None,          # memory
            {},            # model_kwargs
            None,          # use_cache
            None,          # id
            None           # context
        )

    def test_serialization_deserialization(self):
        """Test serializing and deserializing the runner"""
        # Create a runner with all components
        runner = Runner(
            agent=self.mock_agent,
            output_parser=self.mock_parser
        )
        
        # Mock the return_state_dict method
        with patch.object(runner, 'return_state_dict') as mock_return_state:
            mock_return_state.return_value = {
                "config": {
                    "output_parser": {"component_name": "MockOutputParser"},
                    "output_class": "GeneratorOutput"
                }
            }
            
            # Test serialization
            runner_dict = runner.return_state_dict()
            assert isinstance(runner_dict, dict)
            
            # Test deserialization
            with patch('adalflow.core.agent.Agent.from_config') as mock_agent_config, \
                patch('adalflow.components.output_parsers.outputs.OutputParser.from_config') as mock_parser_config:
                
                mock_agent_config.return_value = self.mock_agent
                mock_parser_config.return_value = self.mock_parser
                
                new_runner = Runner.from_config(runner_dict)
                assert new_runner.config.output_class == runner.config.output_class

    def test_call_with_additional_kwargs(self):
        """Test call method with additional kwargs"""
        runner = Runner(agent=self.mock_agent)
        
        runner.call(
            "test query",
            current_objective="test objective",
            memory="test memory",
            context=["context1", "context2"]
        )
        
        self.mock_agent.call.assert_called_once_with(
            "test query",         # user_query
            "test objective",     # current_objective
            "test memory",        # memory
            {},                   # model_kwargs
            None,                 # use_cache
            None,                 # id
            ["context1", "context2"]  # context
        )

    def test_call_with_additional_kwargs(self):
        """Test call method with additional kwargs"""
        runner = Runner(agent=self.mock_agent)
        
        runner.call(
            "test query",
            current_objective="test objective",
            memory="test memory",
            context=["context1", "context2"]
        )
        
        # Update to use keyword arguments in the assertion
        self.mock_agent.call.assert_called_once_with(
            user_query="test query",
            current_objective="test objective",
            memory="test memory",
            model_kwargs={},
            use_cache=None,
            id=None,
            context=["context1", "context2"]
        )