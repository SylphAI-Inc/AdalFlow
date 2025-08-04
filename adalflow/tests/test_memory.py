from adalflow.components.memory.memory import ConversationMemory as Memory
from adalflow.core.types import UserQuery, AssistantResponse
import pytest


def test_empty_memory():
    memory = Memory()
    assert memory() == ""


def test_add_dialog_turn():
    memory = Memory()
    memory.add_dialog_turn("Hello", "Hi! How can I help you?")
    expected = "User:\nquery: Hello\nAssistant: Hi! How can I help you?"
    assert memory() == expected


def test_multiple_turns():
    memory = Memory()
    memory.add_dialog_turn("Hello", "Hi!")
    memory.add_dialog_turn("How are you?", "I'm good!")
    expected = (
        "User:\n"
        "query: Hello\n"
        "Assistant: Hi!\n"
        "User:\n"
        "query: How are you?\n"
        "Assistant: I'm good!"
    )
    assert memory() == expected


def test_add_user_query_and_assistant_response():
    """Test adding user query and assistant response separately."""
    memory = Memory()

    # Add first turn
    turn_id1 = memory.add_user_query("What is Python?")
    assert isinstance(turn_id1, str)
    assert len(turn_id1) > 0

    memory.add_assistant_response("Python is a high-level programming language.")

    # Add second turn
    memory.add_user_query("Is it easy to learn?")
    memory.add_assistant_response(
        "Yes, Python is known for its simple and readable syntax."
    )

    expected = (
        "User:\n"
        "query: What is Python?\n"
        "Assistant: Python is a high-level programming language.\n"
        "User:\n"
        "query: Is it easy to learn?\n"
        "Assistant: Yes, Python is known for its simple and readable syntax."
    )
    assert memory() == expected


def test_add_user_query_with_userquery_object():
    """Test adding user query using UserQuery object."""
    memory = Memory()

    user_query = UserQuery(query_str="Hello from UserQuery object")
    turn_id = memory.add_user_query(user_query)
    assert isinstance(turn_id, str)

    memory.add_assistant_response("Hello! I received your UserQuery object.")
    assert "Hello from UserQuery object" in memory()


def test_add_assistant_response_with_assistantresponse_object():
    """Test adding assistant response using AssistantResponse object."""
    memory = Memory()

    memory.add_user_query("Test with AssistantResponse object")

    assistant_response = AssistantResponse(
        response_str="Response from AssistantResponse object"
    )
    turn_id = memory.add_assistant_response(assistant_response)
    assert isinstance(turn_id, str)

    assert "Response from AssistantResponse object" in memory()


def test_error_double_user_query():
    """Test error when adding user query without completing previous turn."""
    memory = Memory()

    memory.add_user_query("First question")

    with pytest.raises(ValueError) as exc_info:
        memory.add_user_query("Second question")

    assert "already a pending user query" in str(exc_info.value)


def test_error_assistant_response_without_query():
    """Test error when adding assistant response without user query."""
    memory = Memory()

    with pytest.raises(ValueError) as exc_info:
        memory.add_assistant_response("Random response")

    assert "No pending user query found" in str(exc_info.value)


def test_mixed_methods():
    """Test mixing add_dialog_turn with separate methods."""
    memory = Memory()

    # Use add_dialog_turn for first turn
    memory.add_dialog_turn("First question", "First answer")

    # Use separate methods for second turn
    memory.add_user_query("Second question")
    memory.add_assistant_response("Second answer")

    # Use add_dialog_turn again
    memory.add_dialog_turn("Third question", "Third answer")

    expected = (
        "User:\n"
        "query: First question\n"
        "Assistant: First answer\n"
        "User:\n"
        "query: Second question\n"
        "Assistant: Second answer\n"
        "User:\n"
        "query: Third question\n"
        "Assistant: Third answer"
    )
    assert memory() == expected


def test_dialog_turn_ordering():
    """Test that dialog turns maintain correct order."""
    memory = Memory()

    # Add multiple turns
    memory.add_user_query("Question 1")
    memory.add_assistant_response("Answer 1")

    memory.add_user_query("Question 2")
    memory.add_assistant_response("Answer 2")

    memory.add_user_query("Question 3")
    memory.add_assistant_response("Answer 3")

    # Check ordering
    turns = list(memory.current_conversation.dialog_turns.items())
    assert len(turns) == 3
    assert turns[0][0] == 0  # First turn has order 0
    assert turns[1][0] == 1  # Second turn has order 1
    assert turns[2][0] == 2  # Third turn has order 2

    # Verify order field in DialogTurn objects
    assert turns[0][1].order == 0
    assert turns[1][1].order == 1
    assert turns[2][1].order == 2


def test_turn_id_consistency():
    """Test that turn IDs are consistent between add_user_query and add_assistant_response."""
    memory = Memory()

    # Add user query and get turn ID
    turn_id = memory.add_user_query("Test question")

    # Add assistant response and get turn ID
    response_turn_id = memory.add_assistant_response("Test answer")

    # Both should return the same turn ID
    assert turn_id == response_turn_id

    # Verify the turn exists in conversation with this ID
    dialog_turn = None
    for turn in memory.current_conversation.dialog_turns.values():
        if turn.id == turn_id:
            dialog_turn = turn
            break

    assert dialog_turn is not None
    assert dialog_turn.user_query.query_str == "Test question"
    assert dialog_turn.assistant_response.response_str == "Test answer"


def test_memory_with_custom_db():
    """Test memory with custom LocalDB."""
    from adalflow.core.db import LocalDB

    custom_db = LocalDB()
    memory = Memory(turn_db=custom_db)

    memory.add_user_query("Custom DB test")
    memory.add_assistant_response("Custom DB response")

    # Verify data is stored in custom DB
    assert len(custom_db.items) > 0
    output = memory()
    assert "Custom DB test" in output
    assert "Custom DB response" in output


def test_user_query_with_metadata():
    """Test adding user queries with metadata."""
    memory = Memory()

    # Add query with metadata using UserQuery object
    user_query = UserQuery(
        query_str="Tell me about Python",
        metadata={"context": "programming", "level": "beginner"},
    )
    memory.add_user_query(user_query)
    memory.add_assistant_response("Python is a versatile programming language.")

    # Check that metadata appears in output
    output = memory()
    assert "Tell me about Python" in output
    assert "context: programming" in output
    assert "level: beginner" in output


def test_metadata_filtering():
    """Test filtering metadata keys in output."""
    memory = Memory()

    # Add queries with different metadata
    query1 = UserQuery(
        query_str="First question",
        metadata={"public": "visible", "private": "hidden", "context": "test"},
    )
    memory.add_user_query(query1)
    memory.add_assistant_response("First answer")

    query2 = UserQuery(
        query_str="Second question",
        metadata={"public": "also visible", "secret": "should be hidden"},
    )
    memory.add_user_query(query2)
    memory.add_assistant_response("Second answer")

    # Test with filter - only show 'public' and 'context' metadata
    filtered_output = memory(metadata_filter=["public", "context"])

    assert "public: visible" in filtered_output
    assert "public: also visible" in filtered_output
    assert "context: test" in filtered_output

    # These should not appear
    assert "private: hidden" not in filtered_output
    assert "secret: should be hidden" not in filtered_output


def test_mixed_queries_with_and_without_metadata():
    """Test mixing queries with and without metadata."""
    memory = Memory()

    # Regular string query
    memory.add_user_query("Simple question")
    memory.add_assistant_response("Simple answer")

    # Query with metadata
    query_with_meta = UserQuery(
        query_str="Complex question",
        metadata={"source": "documentation", "priority": "high"},
    )
    memory.add_user_query(query_with_meta)
    memory.add_assistant_response("Detailed answer")

    # Another simple query
    memory.add_user_query("Another simple question")
    memory.add_assistant_response("Another answer")

    output = memory()

    # Check all queries are present
    assert "Simple question" in output
    assert "Complex question" in output
    assert "Another simple question" in output

    # Check metadata only appears for the second query
    assert "source: documentation" in output
    assert "priority: high" in output

    # Verify structure
    assert output.count("source:") == 1
    assert output.count("priority:") == 1


def test_empty_metadata():
    """Test UserQuery with empty or None metadata."""
    memory = Memory()

    # UserQuery with None metadata
    query1 = UserQuery(query_str="Query with None metadata", metadata=None)
    memory.add_user_query(query1)
    memory.add_assistant_response("Response 1")

    # UserQuery with empty dict metadata
    query2 = UserQuery(query_str="Query with empty metadata", metadata={})
    memory.add_user_query(query2)
    memory.add_assistant_response("Response 2")

    output = memory()

    # Should contain queries but no metadata lines
    assert "Query with None metadata" in output
    assert "Query with empty metadata" in output
    assert "Response 1" in output
    assert "Response 2" in output

    # Should not have any metadata key: value pairs (only query: should appear)
    # Remove known colons (User:, Assistant:, query:) and check no other colons remain
    cleaned_output = (
        output.replace("User:", "").replace("Assistant:", "").replace("query:", "")
    )
    assert ":" not in cleaned_output


def test_clear_conversation_turns():
    """Test clearing conversation turns."""
    memory = Memory()
    
    # Add some conversation turns
    memory.add_user_query("What is Python?")
    memory.add_assistant_response("Python is a programming language.")
    memory.add_user_query("Tell me more")
    memory.add_assistant_response("It's known for its simplicity.")
    
    # Verify conversation has content
    assert len(memory.current_conversation.dialog_turns) == 2
    assert memory() != ""
    assert "What is Python?" in memory()
    assert "Tell me more" in memory()
    
    # Clear conversation turns
    memory.clear_conversation_turns()
    
    # Verify conversation is cleared
    assert len(memory.current_conversation.dialog_turns) == 0
    assert memory() == ""
    assert memory._pending_user_query is None


def test_clear_conversation_turns_with_pending_query():
    """Test clearing conversation turns when there's a pending user query."""
    memory = Memory()
    
    # Add a complete turn
    memory.add_user_query("First question")
    memory.add_assistant_response("First answer")
    
    # Add a pending user query (no response yet)
    memory.add_user_query("Second question")
    
    # Verify state before clearing
    assert len(memory.current_conversation.dialog_turns) == 1
    assert memory._pending_user_query is not None
    assert memory._pending_user_query["user_query"].query_str == "Second question"
    
    # Clear conversation turns
    memory.clear_conversation_turns()
    
    # Verify everything is cleared including pending query
    assert len(memory.current_conversation.dialog_turns) == 0
    assert memory._pending_user_query is None
    assert memory() == ""


def test_clear_conversation_turns_empty_memory():
    """Test clearing an already empty conversation."""
    memory = Memory()
    
    # Verify memory is empty
    assert len(memory.current_conversation.dialog_turns) == 0
    assert memory() == ""
    
    # Clear empty conversation (should not raise error)
    memory.clear_conversation_turns()
    
    # Still empty
    assert len(memory.current_conversation.dialog_turns) == 0
    assert memory() == ""


def test_add_after_clear():
    """Test adding new conversation turns after clearing."""
    memory = Memory()
    
    # Add initial conversation
    memory.add_user_query("Initial question")
    memory.add_assistant_response("Initial answer")
    
    # Clear
    memory.clear_conversation_turns()
    
    # Add new conversation
    memory.add_user_query("New question")
    memory.add_assistant_response("New answer")
    
    # Verify only new conversation exists
    output = memory()
    assert "New question" in output
    assert "New answer" in output
    assert "Initial question" not in output
    assert "Initial answer" not in output
    assert len(memory.current_conversation.dialog_turns) == 1


def test_clear_preserves_conversation_id():
    """Test that clearing turns preserves the conversation ID."""
    memory = Memory()
    
    # Get initial conversation ID
    initial_conv_id = memory.current_conversation.id
    
    # Add and clear turns
    memory.add_user_query("Test")
    memory.add_assistant_response("Response")
    memory.clear_conversation_turns()
    
    # Conversation ID should remain the same
    assert memory.current_conversation.id == initial_conv_id


def test_clear_does_not_affect_turn_db():
    """Test that clearing conversation turns doesn't affect the turn database."""
    from adalflow.core.db import LocalDB
    
    turn_db = LocalDB()
    memory = Memory(turn_db=turn_db)
    
    # Add turns
    memory.add_user_query("Question 1")
    memory.add_assistant_response("Answer 1")
    memory.add_user_query("Question 2")
    memory.add_assistant_response("Answer 2")
    
    # Verify turn_db has entries
    initial_db_size = len(turn_db.items)
    assert initial_db_size > 0
    
    # Clear conversation turns
    memory.clear_conversation_turns()
    
    # Turn database should still have all entries
    assert len(turn_db.items) == initial_db_size
    
    # But current conversation should be empty
    assert len(memory.current_conversation.dialog_turns) == 0
    assert memory() == ""


def test_clear_with_metadata():
    """Test clearing conversation that contains metadata."""
    memory = Memory()
    
    # Add turns with metadata
    query_with_meta = UserQuery(
        query_str="Question with metadata",
        metadata={"source": "test", "priority": "high"}
    )
    memory.add_user_query(query_with_meta)
    
    response_with_meta = AssistantResponse(
        response_str="Response with metadata",
        metadata={"confidence": "high", "sources": 3}
    )
    memory.add_assistant_response(response_with_meta)
    
    # Verify metadata is present
    output = memory()
    assert "source: test" in output
    assert "priority: high" in output
    
    # Clear and verify
    memory.clear_conversation_turns()
    assert memory() == ""
    assert len(memory.current_conversation.dialog_turns) == 0
