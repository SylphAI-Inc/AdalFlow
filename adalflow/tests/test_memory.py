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
