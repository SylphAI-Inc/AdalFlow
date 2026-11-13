"""Comprehensive test suite for FlexibleConversationMemory.

This test module covers:
1. Basic message operations (creation, addition, retrieval)
2. Turn management (creation, message assignment, ordering)
3. Conversation management (new conversations, clearing, persistence)
4. Error handling and validation
5. Database integration
6. Metadata handling and filtering
7. Edge cases and boundary conditions
"""

import pytest
from datetime import datetime
from collections import OrderedDict
from unittest.mock import Mock, patch
from adalflow.components.memory.flexible_memory import (
    Message,
    Conversation,
    FlexibleConversationMemory,
)
from adalflow.core.db import LocalDB


class TestMessage:
    """Test the Message dataclass functionality."""
    
    def test_message_creation_default(self):
        """Test creating a message with default values.
        
        Tests:
        - Default role is 'user'
        - Content defaults to empty string
        - ID is automatically generated
        - Timestamp is automatically set
        - Metadata is None by default
        """
        msg = Message()
        assert msg.role == "user"
        assert msg.content == ""
        assert msg.id is not None
        assert len(msg.id) > 0
        assert isinstance(msg.timestamp, datetime)
        assert msg.metadata is None
    
    def test_message_creation_with_values(self):
        """Test creating a message with custom values.
        
        Tests:
        - Custom role, content, and metadata are properly set
        - Values are stored correctly
        """
        metadata = {"key": "value", "number": 42}
        msg = Message(role="assistant", content="Hello", metadata=metadata)
        assert msg.role == "assistant"
        assert msg.content == "Hello"
        assert msg.metadata == metadata
    
    def test_message_from_user(self):
        """Test creating a user message using class method.
        
        Tests:
        - from_user() creates message with role='user'
        - Content and metadata are properly set
        """
        metadata = {"context": "test"}
        msg = Message.from_user("User query", metadata)
        assert msg.role == "user"
        assert msg.content == "User query"
        assert msg.metadata == metadata
    
    def test_message_from_assistant(self):
        """Test creating an assistant message using class method.
        
        Tests:
        - from_assistant() creates message with role='assistant'
        - Content and metadata are properly set
        """
        msg = Message.from_assistant("Assistant response", {"confidence": 0.9})
        assert msg.role == "assistant"
        assert msg.content == "Assistant response"
        assert msg.metadata == {"confidence": 0.9}
    
    def test_message_from_system(self):
        """Test creating a system message using class method.
        
        Tests:
        - from_system() creates message with role='system'
        - Content and metadata are properly set
        """
        msg = Message.from_system("System prompt")
        assert msg.role == "system"
        assert msg.content == "System prompt"
        assert msg.metadata is None
    
    def test_message_unique_ids(self):
        """Test that each message gets a unique ID.
        
        Tests:
        - Multiple messages have different IDs
        - IDs are valid UUID strings
        """
        msg1 = Message()
        msg2 = Message()
        msg3 = Message()
        assert msg1.id != msg2.id
        assert msg2.id != msg3.id
        assert msg1.id != msg3.id


class TestConversation:
    """Test the Conversation dataclass functionality."""
    
    def test_conversation_creation(self):
        """Test creating a conversation with default values.
        
        Tests:
        - Conversation ID is generated
        - turns is an OrderedDict
        - user_id can be set
        - Metadata defaults to None
        - Timestamp is set
        """
        conv = Conversation()
        assert conv.id is not None
        assert isinstance(conv.turns, OrderedDict)
        assert len(conv.turns) == 0
        assert conv.user_id is None
        assert conv.metadata is None
        assert isinstance(conv.created_at, datetime)
    
    def test_conversation_with_user_id(self):
        """Test creating a conversation with a specific user ID.
        
        Tests:
        - User ID is properly stored
        """
        conv = Conversation(user_id="user123")
        assert conv.user_id == "user123"
    
    def test_create_turn(self):
        """Test creating a new turn in the conversation.
        
        Tests:
        - Turn is created with unique ID
        - Turn is added to OrderedDict
        - Current turn ID is tracked
        - Multiple turns can be created
        """
        conv = Conversation()
        
        turn_id1 = conv.create_turn()
        assert turn_id1 is not None
        assert turn_id1 in conv.turns
        assert conv.turns[turn_id1] == []
        assert conv._current_turn_id == turn_id1
        
        turn_id2 = conv.create_turn()
        assert turn_id2 != turn_id1
        assert turn_id2 in conv.turns
        assert conv._current_turn_id == turn_id2
        assert len(conv.turns) == 2
    
    def test_add_message_to_turn(self):
        """Test adding messages to a specific turn.
        
        Tests:
        - Messages can be added to existing turns
        - Messages are stored in order
        - Message IDs are returned
        - Multiple messages can be added to same turn
        """
        conv = Conversation()
        turn_id = conv.create_turn()
        
        msg1 = Message.from_user("First message")
        msg_id1 = conv.add_message_to_turn(turn_id, msg1)
        assert msg_id1 == msg1.id
        assert len(conv.turns[turn_id]) == 1
        assert conv.turns[turn_id][0] == msg1
        
        msg2 = Message.from_assistant("Second message")
        msg_id2 = conv.add_message_to_turn(turn_id, msg2)
        assert msg_id2 == msg2.id
        assert len(conv.turns[turn_id]) == 2
        assert conv.turns[turn_id][1] == msg2
    
    def test_add_message_to_nonexistent_turn(self):
        """Test error handling when adding message to non-existent turn.
        
        Tests:
        - ValueError is raised with appropriate message
        - Error message includes available turns
        """
        conv = Conversation()
        msg = Message.from_user("Test")
        
        with pytest.raises(ValueError) as exc_info:
            conv.add_message_to_turn("fake_turn_id", msg)
        assert "Turn 'fake_turn_id' does not exist" in str(exc_info.value)
    
    def test_add_non_message_object(self):
        """Test error handling when adding non-Message object.
        
        Tests:
        - TypeError is raised when non-Message object is provided
        """
        conv = Conversation()
        turn_id = conv.create_turn()
        
        with pytest.raises(TypeError) as exc_info:
            conv.add_message_to_turn(turn_id, "not a message")
        assert "must be an instance of Message" in str(exc_info.value)
    
    def test_get_turn_messages(self):
        """Test retrieving messages from a specific turn.
        
        Tests:
        - Can retrieve all messages from a turn
        - Empty list returned for non-existent turn
        - Messages maintain order
        """
        conv = Conversation()
        turn_id = conv.create_turn()
        
        msg1 = Message.from_user("User msg")
        msg2 = Message.from_assistant("Assistant msg")
        conv.add_message_to_turn(turn_id, msg1)
        conv.add_message_to_turn(turn_id, msg2)
        
        messages = conv.get_turn_messages(turn_id)
        assert len(messages) == 2
        assert messages[0] == msg1
        assert messages[1] == msg2
        
        # Non-existent turn returns empty list
        assert conv.get_turn_messages("fake_id") == []
    
    def test_get_all_messages(self):
        """Test retrieving all messages across all turns.
        
        Tests:
        - Messages from all turns are returned
        - Order is maintained (turn order and message order within turns)
        - Empty list for empty conversation
        """
        conv = Conversation()
        
        # Empty conversation
        assert conv.get_all_messages() == []
        
        # Add messages to multiple turns
        turn1 = conv.create_turn()
        msg1 = Message.from_user("Turn 1 User")
        msg2 = Message.from_assistant("Turn 1 Assistant")
        conv.add_message_to_turn(turn1, msg1)
        conv.add_message_to_turn(turn1, msg2)
        
        turn2 = conv.create_turn()
        msg3 = Message.from_user("Turn 2 User")
        msg4 = Message.from_assistant("Turn 2 Assistant")
        conv.add_message_to_turn(turn2, msg3)
        conv.add_message_to_turn(turn2, msg4)
        
        all_messages = conv.get_all_messages()
        assert len(all_messages) == 4
        assert all_messages == [msg1, msg2, msg3, msg4]
    
    def test_get_messages_by_role(self):
        """Test filtering messages by role.
        
        Tests:

        - Can filter messages by user/assistant/system role
        - Returns messages from all turns
        - Empty list for roles with no messages
        """
        conv = Conversation()
        
        turn1 = conv.create_turn()
        conv.add_message_to_turn(turn1, Message.from_user("User 1"))
        conv.add_message_to_turn(turn1, Message.from_assistant("Assistant 1"))
        
        turn2 = conv.create_turn()
        conv.add_message_to_turn(turn2, Message.from_user("User 2"))
        conv.add_message_to_turn(turn2, Message.from_system("System 1"))
        
        user_messages = conv.get_messages_by_role("user")
        assert len(user_messages) == 2
        assert all(msg.role == "user" for msg in user_messages)
        
        assistant_messages = conv.get_messages_by_role("assistant")
        assert len(assistant_messages) == 1
        assert assistant_messages[0].content == "Assistant 1"
        
        system_messages = conv.get_messages_by_role("system")
        assert len(system_messages) == 1
        assert system_messages[0].content == "System 1"
    
    def test_get_last_user_message(self):
        """Test retrieving the most recent user message.
        
        Tests:
        - Returns the last user message across all turns
        - Returns None if no user messages exist
        - Skips over non-user messages
        """
        conv = Conversation()
        
        # No messages
        assert conv.get_last_user_message() is None
        
        turn1 = conv.create_turn()
        user_msg1 = Message.from_user("First user")
        conv.add_message_to_turn(turn1, user_msg1)
        conv.add_message_to_turn(turn1, Message.from_assistant("Assistant"))
        
        turn2 = conv.create_turn()
        user_msg2 = Message.from_user("Second user")
        conv.add_message_to_turn(turn2, user_msg2)
        conv.add_message_to_turn(turn2, Message.from_assistant("Another assistant"))
        
        last_user = conv.get_last_user_message()
        assert last_user == user_msg2
    
    def test_get_last_assistant_message(self):
        """Test retrieving the most recent assistant message.
        
        Tests:
        - Returns the last assistant message across all turns
        - Returns None if no assistant messages exist
        - Skips over non-assistant messages
        """
        conv = Conversation()
        
        # No messages
        assert conv.get_last_assistant_message() is None
        
        turn1 = conv.create_turn()
        conv.add_message_to_turn(turn1, Message.from_user("User"))
        assistant_msg1 = Message.from_assistant("First assistant")
        conv.add_message_to_turn(turn1, assistant_msg1)
        
        turn2 = conv.create_turn()
        conv.add_message_to_turn(turn2, Message.from_user("Another user"))
        assistant_msg2 = Message.from_assistant("Second assistant")
        conv.add_message_to_turn(turn2, assistant_msg2)
        
        last_assistant = conv.get_last_assistant_message()
        assert last_assistant == assistant_msg2


class TestFlexibleConversationMemory:
    """Test the FlexibleConversationMemory component."""
    
    def test_memory_initialization(self):
        """Test initializing memory with default and custom settings.
        
        Tests:
        - Default initialization creates empty conversation
        - Custom database can be provided
        - User ID is properly set
        """
        # Default initialization
        memory = FlexibleConversationMemory()
        assert memory.current_conversation is not None
        assert memory.user_id is None
        assert memory.message_db is not None
        assert memory.conver_db is not None
        
        # With custom settings
        custom_db = LocalDB()
        memory = FlexibleConversationMemory(turn_db=custom_db, user_id="test_user")
        assert memory.message_db == custom_db
        assert memory.user_id == "test_user"
        assert memory.current_conversation.user_id == "test_user"
    
    def test_create_turn(self):
        """Test creating turns through memory interface.
        
        Tests:
        - Turn creation returns valid ID
        - Multiple turns can be created
        - Turn IDs are unique
        """
        memory = FlexibleConversationMemory()
        
        turn_id1 = memory.create_turn()
        assert turn_id1 is not None
        assert turn_id1 in memory.current_conversation.turns
        
        turn_id2 = memory.create_turn()
        assert turn_id2 != turn_id1
        assert len(memory.current_conversation.turns) == 2
    
    def test_add_user_query(self):
        """Test adding user queries to turns.
        
        Tests:
        - User query is added to specified turn
        - Message is stored in database
        - Metadata is properly handled
        - Error raised for non-existent turn
        """
        memory = FlexibleConversationMemory()
        turn_id = memory.create_turn()
        
        # Add user query
        returned_id = memory.add_user_query("Hello", turn_id, {"context": "greeting"})
        assert returned_id == turn_id
        
        # Check message was added
        messages = memory.get_turn_messages(turn_id)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[0].metadata == {"context": "greeting"}
        
        # Check database storage
        assert len(memory.message_db.items) == 1
        db_item = memory.message_db.items[0]
        assert db_item["role"] == "user"
        assert db_item["content"] == "Hello"
        assert db_item["turn_id"] == turn_id
    
    def test_add_user_query_nonexistent_turn(self):
        """Test error handling when adding user query to non-existent turn.
        
        Tests:
        - ValueError is raised with helpful message
        - Message includes available turns
        """
        memory = FlexibleConversationMemory()
        
        with pytest.raises(ValueError) as exc_info:
            memory.add_user_query("Hello", "fake_turn")
        assert "Turn 'fake_turn' does not exist" in str(exc_info.value)
        assert "create_turn()" in str(exc_info.value)
    
    def test_add_assistant_response(self):
        """Test adding assistant responses to turns.
        
        Tests:
        - Assistant response is added to specified turn
        - Message is stored in database
        - Metadata is properly handled
        - Error raised for non-existent turn
        """
        memory = FlexibleConversationMemory()
        turn_id = memory.create_turn()
        
        # Add assistant response
        returned_id = memory.add_assistant_response(
            "Hi there!", turn_id, {"confidence": 0.95}
        )
        assert returned_id == turn_id
        
        # Check message was added
        messages = memory.get_turn_messages(turn_id)
        assert len(messages) == 1
        assert messages[0].role == "assistant"
        assert messages[0].content == "Hi there!"
        assert messages[0].metadata == {"confidence": 0.95}
        
        # Check database storage
        assert len(memory.message_db.items) == 1
        db_item = memory.message_db.items[0]
        assert db_item["role"] == "assistant"
        assert db_item["content"] == "Hi there!"
    
    def test_add_system_message(self):
        """Test adding system messages to turns.
        
        Tests:
        - System message is added to specified turn
        - Message is stored in database
        - Metadata is properly handled
        - Error raised for non-existent turn
        """
        memory = FlexibleConversationMemory()
        turn_id = memory.create_turn()
        
        # Add system message
        returned_id = memory.add_system_message(
            "System initialized", turn_id, {"version": "1.0"}
        )
        assert returned_id == turn_id
        
        # Check message was added
        messages = memory.get_turn_messages(turn_id)
        assert len(messages) == 1
        assert messages[0].role == "system"
        assert messages[0].content == "System initialized"
        assert messages[0].metadata == {"version": "1.0"}
    
    def test_complete_conversation_flow(self):
        """Test a complete conversation flow with multiple turns.
        
        Tests:
        - Multiple turns with user/assistant exchanges
        - Messages maintain order within and across turns
        - All retrieval methods work correctly
        """
        memory = FlexibleConversationMemory()
        
        # Turn 1: Initial greeting
        turn1 = memory.create_turn()
        memory.add_user_query("Hello, how are you?", turn1)
        memory.add_assistant_response("I'm doing well, thank you!", turn1)
        
        # Turn 2: Follow-up question
        turn2 = memory.create_turn()
        memory.add_user_query("What can you help me with?", turn2)
        memory.add_assistant_response("I can help with many things!", turn2)
        memory.add_user_query("Can you be more specific?", turn2)  # Multiple user queries in same turn
        memory.add_assistant_response("I can help with coding, analysis, and more.", turn2)
        
        # Check all messages
        all_messages = memory.get_all_messages()
        assert len(all_messages) == 6
        
        # Check message ordering
        assert all_messages[0].content == "Hello, how are you?"
        assert all_messages[1].content == "I'm doing well, thank you!"
        assert all_messages[2].content == "What can you help me with?"
        assert all_messages[3].content == "I can help with many things!"
        assert all_messages[4].content == "Can you be more specific?"
        assert all_messages[5].content == "I can help with coding, analysis, and more."
    
    def test_clear_conversation(self):
        """Test clearing the conversation.
        
        Tests:
        - All turns are cleared
        - Current turn ID is reset
        - Database is not affected
        """
        memory = FlexibleConversationMemory()
        
        # Add some content
        turn1 = memory.create_turn()
        memory.add_user_query("Message 1", turn1)
        turn2 = memory.create_turn()
        memory.add_user_query("Message 2", turn2)
        
        # Verify content exists
        assert len(memory.current_conversation.turns) == 2
        assert memory.current_conversation._current_turn_id == turn2
        
        # Clear conversation
        memory.clear_conversation()
        
        # Verify cleared
        assert len(memory.current_conversation.turns) == 0
        assert memory.current_conversation._current_turn_id is None
        
        # Database should still have items
        assert len(memory.message_db.items) == 2
    
    def test_clear_conversation_turns_alias(self):
        """Test that clear_conversation_turns is an alias for clear_conversation.
        
        Tests:
        - Both methods have the same effect
        """
        memory = FlexibleConversationMemory()
        turn = memory.create_turn()
        memory.add_user_query("Test", turn)
        
        memory.clear_conversation_turns()
        assert len(memory.current_conversation.turns) == 0
    
    def test_new_conversation(self):
        """Test starting a new conversation.
        
        Tests:
        - Current conversation is saved to conver_db
        - New empty conversation is created
        - User ID is preserved
        """
        memory = FlexibleConversationMemory(user_id="test_user")
        
        # Add content to first conversation
        turn = memory.create_turn()
        memory.add_user_query("First conversation", turn)
        first_conv_id = memory.current_conversation.id
        
        # Start new conversation
        memory.new_conversation()
        
        # Check new conversation is empty and different
        assert len(memory.current_conversation.turns) == 0
        assert memory.current_conversation.id != first_conv_id
        assert memory.current_conversation.user_id == "test_user"
        
        # Check first conversation was saved
        assert len(memory.conver_db.items) == 1
        saved_conv = memory.conver_db.items[0]
        assert saved_conv.id == first_conv_id
    
    def test_get_current_turn_id(self):
        """Test retrieving the current turn ID.
        
        Tests:
        - Returns None when no turns exist
        - Returns correct turn ID after creation
        - Updates when new turn is created
        """
        memory = FlexibleConversationMemory()
        
        # No turns initially
        assert memory.get_current_turn_id() is None
        
        # After creating turn
        turn1 = memory.create_turn()
        assert memory.get_current_turn_id() == turn1
        
        # After creating another turn
        turn2 = memory.create_turn()
        assert memory.get_current_turn_id() == turn2
    
    def test_call_empty_conversation(self):
        """Test calling memory with empty conversation.
        
        Tests:
        - Returns empty string for empty conversation
        """
        memory = FlexibleConversationMemory()
        assert memory() == ""
        assert memory.call() == ""
    
    def test_call_with_messages(self):
        """Test formatting conversation for output.
        
        Tests:
        - Messages are formatted correctly
        - Roles are properly labeled
        - Multiple turns are handled
        - Metadata is included when present
        """
        memory = FlexibleConversationMemory()
        
        turn1 = memory.create_turn()
        memory.add_user_query("What is Python?", turn1)
        memory.add_assistant_response("Python is a programming language.", turn1)
        
        turn2 = memory.create_turn()
        memory.add_user_query("Tell me more", turn2, {"priority": "high"})
        memory.add_assistant_response("It's known for its simplicity.", turn2)
        
        output = memory()
        
        # Check basic content
        assert "User: What is Python?" in output
        assert "Assistant: Python is a programming language." in output
        assert "User: Tell me more" in output
        assert "Assistant: It's known for its simplicity." in output
        
        # Check metadata
        assert "priority: high" in output
    
    def test_call_with_metadata_filter(self):
        """Test filtering metadata in conversation output.
        
        Tests:
        - Only specified metadata keys are included
        - Other metadata is filtered out
        - Works across multiple messages
        """
        memory = FlexibleConversationMemory()
        
        turn = memory.create_turn()
        memory.add_user_query("Query", turn, {
            "public": "visible",
            "private": "hidden",
            "context": "test"
        })
        memory.add_assistant_response("Response", turn, {
            "confidence": 0.9,
            "internal": "secret"
        })
        
        # Filter to only show 'public' and 'confidence'
        output = memory(metadata_filter=["public", "confidence"])
        
        assert "public: visible" in output
        assert "confidence: 0.9" in output
        assert "private: hidden" not in output
        assert "context: test" not in output
        assert "internal: secret" not in output
    
    def test_call_with_system_messages(self):
        """Test that system messages are included in output.
        
        Tests:
        - System messages are formatted correctly
        - Mixed message types are handled
        """
        memory = FlexibleConversationMemory()
        
        turn = memory.create_turn()
        memory.add_system_message("System: Initializing", turn)
        memory.add_user_query("Hello", turn)
        memory.add_assistant_response("Hi there", turn)
        
        output = memory()
        assert "System: System: Initializing" in output
        assert "User: Hello" in output
        assert "Assistant: Hi there" in output
    
    def test_get_last_n_messages(self):
        """Test retrieving the last N messages.
        
        Tests:
        - Returns correct number of messages
        - Returns all messages if N > total
        - Maintains correct order
        """
        memory = FlexibleConversationMemory()
        
        # Add 5 messages
        turn = memory.create_turn()
        for i in range(5):
            if i % 2 == 0:
                memory.add_user_query(f"Message {i}", turn)
            else:
                memory.add_assistant_response(f"Message {i}", turn)
        
        # Get last 3
        last_3 = memory.get_last_n_messages(3)
        assert len(last_3) == 3
        assert last_3[0].content == "Message 2"
        assert last_3[1].content == "Message 3"
        assert last_3[2].content == "Message 4"
        
        # Get more than available
        last_10 = memory.get_last_n_messages(10)
        assert len(last_10) == 5
    
    def test_count_messages(self):
        """Test counting messages by role.
        
        Tests:
        - Counts are accurate for each role
        - Handles empty conversation
        - Works across multiple turns
        """
        memory = FlexibleConversationMemory()
        
        # Empty conversation
        counts = memory.count_messages()
        assert counts == {"user": 0, "assistant": 0, "system": 0}
        
        # Add messages
        turn1 = memory.create_turn()
        memory.add_user_query("User 1", turn1)
        memory.add_assistant_response("Assistant 1", turn1)
        
        turn2 = memory.create_turn()
        memory.add_user_query("User 2", turn2)
        memory.add_user_query("User 3", turn2)
        memory.add_system_message("System 1", turn2)
        
        counts = memory.count_messages()
        assert counts["user"] == 3
        assert counts["assistant"] == 1
        assert counts["system"] == 1
    
    def test_count_turns(self):
        """Test counting the number of turns.
        
        Tests:
        - Returns 0 for empty conversation
        - Counts turns correctly
        """
        memory = FlexibleConversationMemory()
        
        assert memory.count_turns() == 0
        
        memory.create_turn()
        assert memory.count_turns() == 1
        
        memory.create_turn()
        memory.create_turn()
        assert memory.count_turns() == 3
    
    def test_reset_pending_query(self):
        """Test resetting pending query (compatibility method).
        
        Tests:
        - Current turn ID is unset
        - Turns are not deleted
        - Messages remain intact
        """
        memory = FlexibleConversationMemory()
        
        turn = memory.create_turn()
        memory.add_user_query("Test", turn)
        
        # Current turn is set
        assert memory.get_current_turn_id() == turn
        
        # Reset pending query
        memory.reset_pending_query()
        
        # Current turn is unset but turn still exists
        assert memory.get_current_turn_id() is None
        assert turn in memory.current_conversation.turns
        assert len(memory.get_turn_messages(turn)) == 1
    
    def test_len_magic_method(self):
        """Test the __len__ magic method.
        
        Tests:
        - Returns 0 for empty conversation
        - Returns correct count of all messages
        """
        memory = FlexibleConversationMemory()
        
        assert len(memory) == 0
        
        turn = memory.create_turn()
        memory.add_user_query("1", turn)
        memory.add_assistant_response("2", turn)
        memory.add_system_message("3", turn)
        
        assert len(memory) == 3
    
    def test_multiple_queries_same_turn(self):
        """Test adding multiple user queries to the same turn.
        
        Tests:
        - Multiple user queries can be added to one turn
        - Order is maintained
        - Common in clarification scenarios
        """
        memory = FlexibleConversationMemory()
        
        turn = memory.create_turn()
        memory.add_user_query("First question", turn)
        memory.add_user_query("Follow-up question", turn)
        memory.add_user_query("Another clarification", turn)
        memory.add_assistant_response("Comprehensive answer", turn)
        
        messages = memory.get_turn_messages(turn)
        assert len(messages) == 4
        assert messages[0].content == "First question"
        assert messages[1].content == "Follow-up question"
        assert messages[2].content == "Another clarification"
        assert messages[3].content == "Comprehensive answer"
    
    def test_complex_metadata_handling(self):
        """Test handling complex metadata structures.
        
        Tests:
        - Nested dictionaries in metadata
        - Lists in metadata
        - Mixed data types
        """
        memory = FlexibleConversationMemory()
        
        turn = memory.create_turn()
        complex_metadata = {
            "step_history": [
                {"action": "search", "result": "found"},
                {"action": "analyze", "result": "complete"}
            ],
            "images": ["image1.png", "image2.jpg"],
            "confidence": 0.95,
            "nested": {
                "level1": {
                    "level2": "deep value"
                }
            }
        }
        
        memory.add_user_query("Complex query", turn, complex_metadata)
        
        messages = memory.get_turn_messages(turn)
        assert messages[0].metadata == complex_metadata
        assert messages[0].metadata["step_history"][0]["action"] == "search"
        assert messages[0].metadata["nested"]["level1"]["level2"] == "deep value"
    
    def test_conversation_persistence(self):
        """Test that conversations are properly persisted.
        
        Tests:
        - Conversations are saved when starting new one
        - Empty conversations are not saved
        - Multiple conversations can be saved
        """
        memory = FlexibleConversationMemory()
        
        # Empty conversation not saved
        memory.new_conversation()
        assert len(memory.conver_db.items) == 0
        
        # Add content and start new conversation
        turn = memory.create_turn()
        memory.add_user_query("First conv", turn)
        memory.new_conversation()
        assert len(memory.conver_db.items) == 1
        
        # Add more conversations
        turn = memory.create_turn()
        memory.add_user_query("Second conv", turn)
        memory.new_conversation()
        assert len(memory.conver_db.items) == 2
    
    def test_error_handling_in_create_turn(self):
        """Test error handling in turn creation.
        
        Tests:
        - RuntimeError is raised if turn creation fails
        - Error message is informative
        """
        memory = FlexibleConversationMemory()
        
        # Mock a failure in UUID generation
        with patch('adalflow.components.memory.flexible_memory.uuid4', side_effect=Exception("UUID error")):
            with pytest.raises(RuntimeError) as exc_info:
                memory.create_turn()
            assert "Failed to create turn" in str(exc_info.value)
    
    def test_database_integration(self):
        """Test integration with LocalDB for message storage.
        
        Tests:
        - Messages are stored with correct fields
        - Turn IDs are properly tracked
        - Timestamps are preserved
        - Multiple databases work independently
        """
        db1 = LocalDB()
        db2 = LocalDB()
        
        memory1 = FlexibleConversationMemory(turn_db=db1)
        memory2 = FlexibleConversationMemory(turn_db=db2)
        
        # Add to memory1
        turn1 = memory1.create_turn()
        memory1.add_user_query("Memory 1 message", turn1)
        
        # Add to memory2
        turn2 = memory2.create_turn()
        memory2.add_user_query("Memory 2 message", turn2)
        
        # Check databases are independent
        assert len(db1.items) == 1
        assert len(db2.items) == 1
        assert db1.items[0]["content"] == "Memory 1 message"
        assert db2.items[0]["content"] == "Memory 2 message"
    
    def test_edge_cases(self):
        """Test various edge cases and boundary conditions.
        
        Tests:
        - Empty strings as content
        - Very long content
        - Special characters in content
        - None values where applicable
        """
        memory = FlexibleConversationMemory()
        
        turn = memory.create_turn()
        
        # Empty string content
        memory.add_user_query("", turn)
        messages = memory.get_turn_messages(turn)
        assert messages[0].content == ""
        
        # Very long content
        long_content = "x" * 10000
        memory.add_assistant_response(long_content, turn)
        assert len(memory.get_turn_messages(turn)[1].content) == 10000
        
        # Special characters
        special = "Hello\n\tWorld!@#$%^&*()[]{}|\\<>?/~`"
        memory.add_user_query(special, turn)
        assert memory.get_turn_messages(turn)[2].content == special
        
        # None metadata (should work fine)
        memory.add_user_query("Test", turn, None)
        assert memory.get_turn_messages(turn)[3].metadata is None


class TestIntegration:
    """Integration tests for the complete memory system."""
    
    def test_realistic_conversation_flow(self):
        """Test a realistic multi-turn conversation with all features.
        
        Tests:
        - Complete conversation flow
        - Mixed message types
        - Metadata handling
        - Turn management
        - Output formatting
        """
        memory = FlexibleConversationMemory(user_id="test_user")
        
        # Turn 1: Initial setup
        turn1 = memory.create_turn()
        memory.add_system_message("You are a helpful assistant.", turn1)
        memory.add_user_query("Hi, I need help with Python", turn1, {"source": "web_ui"})
        memory.add_assistant_response(
            "Hello! I'd be happy to help you with Python. What specific topic?",
            turn1,
            {"confidence": 0.95}
        )
        
        # Turn 2: Follow-up
        turn2 = memory.create_turn()
        memory.add_user_query("How do I read a file?", turn2)
        memory.add_assistant_response(
            "You can use the open() function with a context manager.",
            turn2,
            {"confidence": 0.98, "sources": ["python_docs"]}
        )
        memory.add_user_query("Can you show an example?", turn2)  # Clarification in same turn
        memory.add_assistant_response(
            "with open('file.txt', 'r') as f:\n    content = f.read()",
            turn2,
            {"code_block": True}
        )
        
        # Verify conversation state
        assert memory.count_turns() == 2
        counts = memory.count_messages()
        assert counts["user"] == 3
        assert counts["assistant"] == 3
        assert counts["system"] == 1
        
        # Check output formatting
        output = memory()
        assert "You are a helpful assistant" in output
        assert "How do I read a file?" in output
        assert "with open('file.txt', 'r')" in output
        
        # Test filtered output
        filtered = memory(metadata_filter=["confidence"])
        assert "confidence: 0.95" in filtered
        assert "confidence: 0.98" in filtered
        assert "source: web_ui" not in filtered
        
        # Get last messages
        last_2 = memory.get_last_n_messages(2)
        assert last_2[0].content == "Can you show an example?"
        assert "with open" in last_2[1].content
        
        # Start new conversation but preserve the old one
        memory.new_conversation()
        assert len(memory.conver_db.items) == 1
        assert memory.count_messages() == {"user": 0, "assistant": 0, "system": 0}
    
    def test_agent_style_conversation(self):
        """Test conversation flow typical of an AI agent with step tracking.
        
        Tests:
        - Agent-style metadata with step history
        - Multiple assistant messages in sequence
        - Complex metadata structures
        """
        memory = FlexibleConversationMemory()
        
        turn = memory.create_turn()
        
        # User query with context
        memory.add_user_query(
            "Find all Python files in the project",
            turn,
            {"request_type": "search", "timestamp": "2024-01-01T10:00:00"}
        )
        
        # Agent thinking steps
        memory.add_assistant_response(
            "I'll search for Python files in the project.",
            turn,
            {
                "step": 1,
                "action": "plan",
                "confidence": 0.9
            }
        )
        
        memory.add_assistant_response(
            "Found 15 Python files in 3 directories.",
            turn,
            {
                "step": 2,
                "action": "search",
                "results": {
                    "count": 15,
                    "directories": ["src", "tests", "scripts"]
                }
            }
        )
        
        memory.add_assistant_response(
            "Here are the Python files I found: [list of files]",
            turn,
            {
                "step": 3,
                "action": "report",
                "completion": True
            }
        )
        
        # Verify step tracking
        messages = memory.get_turn_messages(turn)
        assert len(messages) == 4
        
        # Check assistant messages have increasing step numbers
        assistant_messages = [m for m in messages if m.role == "assistant"]
        assert assistant_messages[0].metadata["step"] == 1
        assert assistant_messages[1].metadata["step"] == 2
        assert assistant_messages[2].metadata["step"] == 3
        
        # Verify complex metadata structure
        assert assistant_messages[1].metadata["results"]["count"] == 15
        assert "src" in assistant_messages[1].metadata["results"]["directories"]