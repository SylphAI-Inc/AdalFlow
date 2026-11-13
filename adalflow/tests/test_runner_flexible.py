"""Test suite for RunnerFlexible with FlexibleConversationMemory integration.

This module tests:
1. Runner initialization with flexible memory
2. Safe memory operations that never fail the runner
3. Turn management during execution
4. Error handling and recovery
5. Memory persistence across multiple runs
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from adalflow.components.agent.runner_flexible import RunnerFlexible
from adalflow.components.memory.flexible_memory import FlexibleConversationMemory
from adalflow.components.agent.agent import Agent
from adalflow.core.types import (
    GeneratorOutput,
    Function,
    FunctionOutput,
    RunnerResult,
    UserQuery,
    AssistantResponse,
    ToolOutput,
)


class TestRunnerFlexibleMemoryIntegration:
    """Test RunnerFlexible with FlexibleConversationMemory."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock(spec=Agent)
        agent.max_steps = 3
        agent.answer_data_type = str
        agent.is_thinking_model = False
        
        # Mock planner
        agent.planner = Mock()
        agent.planner.call = Mock()
        agent.planner.acall = Mock()
        agent.planner.get_prompt = Mock(return_value="test prompt")
        agent.planner.estimated_token_count = 100
        
        # Mock tool manager
        agent.tool_manager = Mock()
        agent.tool_manager.tools = []
        agent.tool_manager.execute_func = Mock()
        agent.tool_manager.execute_func_async = Mock()
        
        return agent
    
    @pytest.fixture
    def flexible_memory(self):
        """Create a flexible memory instance."""
        return FlexibleConversationMemory()
    
    @pytest.fixture
    def runner_with_memory(self, mock_agent, flexible_memory):
        """Create a runner with flexible memory."""
        return RunnerFlexible(
            agent=mock_agent,
            conversation_memory=flexible_memory
        )
    
    def test_runner_initialization_with_memory(self, mock_agent, flexible_memory):
        """Test that runner initializes correctly with flexible memory.
        
        Tests:
        - Runner accepts FlexibleConversationMemory
        - use_conversation_memory flag is set correctly
        - Safe memory operation methods are available
        """
        runner = RunnerFlexible(agent=mock_agent, conversation_memory=flexible_memory)
        
        assert runner.conversation_memory == flexible_memory
        assert runner.use_conversation_memory is True
        assert hasattr(runner, '_safe_create_turn')
        assert hasattr(runner, '_safe_add_user_query')
        assert hasattr(runner, '_safe_add_assistant_response')
        assert hasattr(runner, '_safe_get_conversation_history')
    
    def test_safe_create_turn(self, runner_with_memory):
        """Test safe turn creation that never fails.
        
        Tests:
        - Normal turn creation works
        - Returns None when memory fails
        - Logs warning on failure
        """
        # Normal operation
        turn_id = runner_with_memory._safe_create_turn()
        assert turn_id is not None
        assert isinstance(turn_id, str)
        
        # Test failure handling
        runner_with_memory.conversation_memory.create_turn = Mock(
            side_effect=Exception("Memory error")
        )
        
        with patch('adalflow.components.agent.runner_flexible.log') as mock_log:
            result = runner_with_memory._safe_create_turn()
            assert result is None
            mock_log.warning.assert_called_once()
            assert "Failed to create turn" in str(mock_log.warning.call_args)
    
    def test_safe_add_user_query(self, runner_with_memory):
        """Test safe user query addition.
        
        Tests:
        - Adding string queries
        - Adding UserQuery objects
        - Handling memory failures gracefully
        """
        turn_id = runner_with_memory._safe_create_turn()
        
        # Test string query
        result = runner_with_memory._safe_add_user_query(
            "Test query", turn_id, {"meta": "data"}
        )
        assert result == turn_id
        
        # Test UserQuery object
        user_query = UserQuery(query_str="Object query", metadata={"key": "value"})
        result = runner_with_memory._safe_add_user_query(user_query, turn_id)
        assert result == turn_id
        
        # Test with None turn_id
        result = runner_with_memory._safe_add_user_query("Query", None)
        assert result is None
        
        # Test failure handling
        runner_with_memory.conversation_memory.add_user_query = Mock(
            side_effect=Exception("Add failed")
        )
        
        with patch('adalflow.components.agent.runner_flexible.log') as mock_log:
            result = runner_with_memory._safe_add_user_query("Query", turn_id)
            assert result is None
            mock_log.warning.assert_called_once()
    
    def test_safe_add_assistant_response(self, runner_with_memory):
        """Test safe assistant response addition.
        
        Tests:
        - Adding string responses
        - Adding AssistantResponse objects
        - Handling memory failures gracefully
        """
        turn_id = runner_with_memory._safe_create_turn()
        
        # Test string response
        result = runner_with_memory._safe_add_assistant_response(
            "Test response", turn_id, {"meta": "data"}
        )
        assert result == turn_id
        
        # Test AssistantResponse object
        assistant_response = AssistantResponse(
            response_str="Object response", 
            metadata={"key": "value"}
        )
        result = runner_with_memory._safe_add_assistant_response(assistant_response, turn_id)
        assert result == turn_id
        
        # Test with None turn_id
        result = runner_with_memory._safe_add_assistant_response("Response", None)
        assert result is None
        
        # Test failure handling
        runner_with_memory.conversation_memory.add_assistant_response = Mock(
            side_effect=Exception("Add failed")
        )
        
        with patch('adalflow.components.agent.runner_flexible.log') as mock_log:
            result = runner_with_memory._safe_add_assistant_response("Response", turn_id)
            assert result is None
            mock_log.warning.assert_called_once()
    
    def test_safe_get_conversation_history(self, runner_with_memory):
        """Test safe conversation history retrieval.
        
        Tests:
        - Normal history retrieval
        - Returns empty string on failure
        - Never raises exceptions
        """
        # Add some conversation
        turn_id = runner_with_memory._safe_create_turn()
        runner_with_memory._safe_add_user_query("Hello", turn_id)
        runner_with_memory._safe_add_assistant_response("Hi there", turn_id)
        
        # Get history
        history = runner_with_memory._safe_get_conversation_history()
        assert "Hello" in history
        assert "Hi there" in history
        
        # Test failure handling - mock the call method directly
        def mock_call():
            raise Exception("History error")
        
        original_call = runner_with_memory.conversation_memory.call
        runner_with_memory.conversation_memory.call = mock_call
        # Also need to replace __call__ since Python uses that
        original_dunder_call = runner_with_memory.conversation_memory.__class__.__call__
        runner_with_memory.conversation_memory.__class__.__call__ = lambda self: mock_call()
        
        try:
            with patch('adalflow.components.agent.runner_flexible.log') as mock_log:
                result = runner_with_memory._safe_get_conversation_history()
                assert result == ""
                mock_log.warning.assert_called_once()
        finally:
            # Restore original methods
            runner_with_memory.conversation_memory.call = original_call
            runner_with_memory.conversation_memory.__class__.__call__ = original_dunder_call
    
    def test_runner_continues_on_memory_failure(self, mock_agent, flexible_memory):
        """Test that runner continues execution even when memory operations fail.
        
        Tests:
        - Runner completes successfully despite memory errors
        - Warnings are logged but execution continues
        - Final result is still produced
        """
        runner = RunnerFlexible(agent=mock_agent, conversation_memory=flexible_memory)
        
        # Make all memory operations fail
        flexible_memory.create_turn = Mock(side_effect=Exception("Memory failed"))
        flexible_memory.add_user_query = Mock(side_effect=Exception("Memory failed"))
        flexible_memory.add_assistant_response = Mock(side_effect=Exception("Memory failed"))
        flexible_memory.__call__ = Mock(side_effect=Exception("Memory failed"))
        
        # Setup successful agent execution
        mock_function = Function(name="finish")
        mock_function._is_answer_final = True
        mock_function._answer = "Final answer"
        
        mock_agent.planner.call.return_value = GeneratorOutput(
            data=mock_function,
            error=None,
            raw_response="raw"
        )
        
        # Run should complete successfully despite memory failures
        with patch('adalflow.components.agent.runner_flexible.log') as mock_log:
            result = runner.call({"input_str": "Test query"})
            
            # Check runner completed successfully
            assert isinstance(result, RunnerResult)
            assert result.answer == "Final answer"
            
            # Check warnings were logged for memory failures
            warning_calls = mock_log.warning.call_args_list
            assert len(warning_calls) > 0
            assert any("Failed to" in str(call) for call in warning_calls)
    
    def test_async_runner_with_memory(self, mock_agent, flexible_memory):
        """Test async runner execution with memory.
        
        Tests:
        - Async execution works with memory
        - Turn management in async context
        - Memory operations are safe in async
        """
        runner = RunnerFlexible(agent=mock_agent, conversation_memory=flexible_memory)
        
        # Setup async mock
        mock_function = Function(name="finish")
        mock_function._is_answer_final = True
        mock_function._answer = "Async answer"
        
        async def mock_acall(*args, **kwargs):
            return GeneratorOutput(
                data=mock_function,
                error=None,
                raw_response="raw"
            )
        
        mock_agent.planner.acall = mock_acall
        
        # Run async
        async def run_test():
            result = await runner.acall({"input_str": "Async query"})
            return result
        
        result = asyncio.run(run_test())
        
        assert isinstance(result, RunnerResult)
        assert result.answer == "Async answer"
        
        # Check memory has content
        history = flexible_memory()
        assert "Async query" in history
        assert "Async answer" in history
    
    def test_memory_persistence_across_runs(self, runner_with_memory, mock_agent):
        """Test that memory persists across multiple runner executions.
        
        Tests:
        - Memory accumulates across runs
        - Each run creates a new turn
        - History is available in subsequent runs
        """
        # Setup mock responses
        def create_mock_response(answer):
            mock_function = Function(name="finish")
            mock_function._is_answer_final = True
            mock_function._answer = answer
            return GeneratorOutput(data=mock_function, error=None, raw_response="raw")
        
        # First run
        mock_agent.planner.call.return_value = create_mock_response("Answer 1")
        result1 = runner_with_memory.call({"input_str": "Query 1"})
        assert result1.answer == "Answer 1"
        
        # Second run
        mock_agent.planner.call.return_value = create_mock_response("Answer 2")
        result2 = runner_with_memory.call({"input_str": "Query 2"})
        assert result2.answer == "Answer 2"
        
        # Check memory has both conversations
        history = runner_with_memory.conversation_memory()
        assert "Query 1" in history
        assert "Answer 1" in history
        assert "Query 2" in history
        assert "Answer 2" in history
        
        # Check we have 2 turns
        assert runner_with_memory.conversation_memory.count_turns() == 2
    
    def test_memory_with_complex_metadata(self, runner_with_memory, mock_agent):
        """Test memory handling of complex metadata.
        
        Tests:
        - Step history is properly stored
        - Complex nested metadata works
        - Metadata survives memory errors
        """
        # Setup mock with multiple steps
        mock_function1 = Function(name="search", args=["query"])
        mock_agent.planner.call.side_effect = [
            GeneratorOutput(data=mock_function1, error=None, raw_response="raw"),
            GeneratorOutput(
                data=Function(name="finish", _is_answer_final=True, _answer="Done"),
                error=None,
                raw_response="raw"
            )
        ]
        
        mock_agent.tool_manager.execute_func.return_value = FunctionOutput(
            name="search",
            input=mock_function1,
            output=ToolOutput(output="Search results", observation="Found items")
        )
        
        # Run with multiple steps
        result = runner_with_memory.call({"input_str": "Multi-step query"})
        
        # Check step history in metadata
        all_messages = runner_with_memory.conversation_memory.get_all_messages()
        assistant_messages = [m for m in all_messages if m.role == "assistant"]
        
        if assistant_messages:
            last_assistant = assistant_messages[-1]
            assert last_assistant.metadata is not None
            assert "step_history" in last_assistant.metadata
            assert len(last_assistant.metadata["step_history"]) > 0
    
    def test_runner_without_memory(self, mock_agent):
        """Test that runner works fine without memory.
        
        Tests:
        - Runner can be initialized without memory
        - All safe memory operations handle None memory
        - Execution completes normally
        """
        runner = RunnerFlexible(agent=mock_agent, conversation_memory=None)
        
        assert runner.conversation_memory is None
        assert runner.use_conversation_memory is False
        
        # All safe operations should return appropriate defaults
        assert runner._safe_create_turn() is None
        assert runner._safe_add_user_query("query", "turn_id") is None
        assert runner._safe_add_assistant_response("response", "turn_id") is None
        assert runner._safe_get_conversation_history() == ""
        
        # Runner should work normally
        mock_function = Function(name="finish")
        mock_function._is_answer_final = True
        mock_function._answer = "No memory answer"
        
        mock_agent.planner.call.return_value = GeneratorOutput(
            data=mock_function,
            error=None,
            raw_response="raw"
        )
        
        result = runner.call({"input_str": "Test"})
        assert result.answer == "No memory answer"


class TestMemoryErrorRecovery:
    """Test error recovery scenarios with memory failures."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock(spec=Agent)
        agent.max_steps = 3
        agent.answer_data_type = str
        agent.is_thinking_model = False
        
        # Mock planner
        agent.planner = Mock()
        agent.planner.call = Mock()
        agent.planner.acall = Mock()
        agent.planner.get_prompt = Mock(return_value="test prompt")
        agent.planner.estimated_token_count = 100
        
        # Mock tool manager
        agent.tool_manager = Mock()
        agent.tool_manager.tools = []
        agent.tool_manager.execute_func = Mock()
        agent.tool_manager.execute_func_async = Mock()
        
        return agent
    
    def test_memory_failure_during_turn_creation(self, mock_agent):
        """Test recovery when turn creation fails.
        
        Tests:
        - Runner continues without turn_id
        - Subsequent operations handle None turn_id
        - Execution completes successfully
        """
        memory = FlexibleConversationMemory()
        memory.create_turn = Mock(side_effect=Exception("Turn creation failed"))
        
        runner = RunnerFlexible(agent=mock_agent, conversation_memory=memory)
        
        mock_function = Function(name="finish", _is_answer_final=True, _answer="Success")
        mock_agent.planner.call.return_value = GeneratorOutput(
            data=mock_function, error=None, raw_response="raw"
        )
        
        with patch('adalflow.components.agent.runner_flexible.log'):
            result = runner.call({"input_str": "Test"})
            assert result.answer == "Success"
    
    def test_partial_memory_failure(self, mock_agent):
        """Test when some memory operations fail but others succeed.
        
        Tests:
        - Turn creation succeeds
        - User query fails
        - Assistant response fails
        - Runner still completes
        """
        memory = FlexibleConversationMemory()
        original_create = memory.create_turn
        memory.add_user_query = Mock(side_effect=Exception("User query failed"))
        memory.add_assistant_response = Mock(side_effect=Exception("Assistant response failed"))
        
        runner = RunnerFlexible(agent=mock_agent, conversation_memory=memory)
        
        mock_function = Function(name="finish", _is_answer_final=True, _answer="Partial success")
        mock_agent.planner.call.return_value = GeneratorOutput(
            data=mock_function, error=None, raw_response="raw"
        )
        
        with patch('adalflow.components.agent.runner_flexible.log'):
            result = runner.call({"input_str": "Test"})
            assert result.answer == "Partial success"
            
            # Turn should have been created successfully
            assert memory.count_turns() == 1
    
    def test_memory_recovery_between_runs(self, mock_agent):
        """Test that memory can recover between runs.
        
        Tests:
        - First run with memory failure
        - Memory recovers
        - Second run succeeds with memory
        """
        memory = FlexibleConversationMemory()
        runner = RunnerFlexible(agent=mock_agent, conversation_memory=memory)
        
        mock_function = Function(name="finish", _is_answer_final=True, _answer="Answer")
        mock_agent.planner.call.return_value = GeneratorOutput(
            data=mock_function, error=None, raw_response="raw"
        )
        
        # First run with failing memory
        original_create = memory.create_turn
        memory.create_turn = Mock(side_effect=Exception("Temporary failure"))
        
        with patch('adalflow.components.agent.runner_flexible.log'):
            result1 = runner.call({"input_str": "Query 1"})
            assert result1.answer == "Answer"
        
        # Restore memory functionality
        memory.create_turn = original_create
        
        # Second run should work with memory
        result2 = runner.call({"input_str": "Query 2"})
        assert result2.answer == "Answer"
        
        # Second run should have created a turn
        assert memory.count_turns() == 1
        history = memory()
        assert "Query 2" in history