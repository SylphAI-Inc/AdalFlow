"""Test the Jinja2 template rendering in FlexibleConversationMemory.

This module specifically tests:
1. Template rendering with call() method
2. Metadata filtering in templates
3. Multiple turns and messages formatting
4. System, user, and assistant message formatting
5. Edge cases in template rendering
"""

import pytest
from adalflow.components.memory.flexible_memory import (
    Message,
    Conversation,
    FlexibleConversationMemory,
    CONVERSATION_TEMPLATE,
)
from adalflow.core.prompt_builder import Prompt


class TestFlexibleMemoryTemplateRendering:
    """Test the Jinja2 template rendering functionality."""
    
    def test_basic_template_rendering(self):
        """Test basic conversation rendering with the template.
        
        Tests:
        - Single turn with user and assistant messages
        - Proper formatting with "User:" and "Assistant:" prefixes
        - Newline handling
        """
        memory = FlexibleConversationMemory()
        
        turn_id = memory.create_turn()
        memory.add_user_query("Hello, how are you?", turn_id)
        memory.add_assistant_response("I'm doing well, thank you!", turn_id)
        
        # Call the memory to render the template
        output = memory.call()
        
        # Check the formatted output
        assert "User: Hello, how are you?" in output
        assert "Assistant: I'm doing well, thank you!" in output
        
        # Check order
        lines = output.strip().split('\n')
        user_line_idx = next(i for i, line in enumerate(lines) if "User:" in line)
        assistant_line_idx = next(i for i, line in enumerate(lines) if "Assistant:" in line)
        assert user_line_idx < assistant_line_idx
    
    def test_multiple_turns_rendering(self):
        """Test rendering multiple conversation turns.
        
        Tests:
        - Multiple turns are rendered in order
        - Each turn's messages are grouped together
        - Proper spacing between turns
        """
        memory = FlexibleConversationMemory()
        
        # Turn 1
        turn1 = memory.create_turn()
        memory.add_user_query("What is Python?", turn1)
        memory.add_assistant_response("Python is a programming language.", turn1)
        
        # Turn 2
        turn2 = memory.create_turn()
        memory.add_user_query("What can I do with it?", turn2)
        memory.add_assistant_response("You can build web apps, analyze data, and more.", turn2)
        
        output = memory.call()
        
        # Check all messages are present
        assert "What is Python?" in output
        assert "Python is a programming language" in output
        assert "What can I do with it?" in output
        assert "You can build web apps" in output
        
        # Check order is maintained
        python_idx = output.index("What is Python?")
        python_answer_idx = output.index("Python is a programming")
        what_can_idx = output.index("What can I do with it?")
        web_apps_idx = output.index("You can build web apps")
        
        assert python_idx < python_answer_idx < what_can_idx < web_apps_idx
    
    def test_metadata_rendering(self):
        """Test metadata rendering in the template.
        
        Tests:
        - Metadata is rendered below the message
        - Metadata keys and values are formatted correctly
        - Multiple metadata items are shown
        """
        memory = FlexibleConversationMemory()
        
        turn = memory.create_turn()
        memory.add_user_query(
            "Search for Python tutorials",
            turn,
            metadata={
                "source": "web_interface",
                "priority": "high",
                "timestamp": "2024-01-01T10:00:00"
            }
        )
        memory.add_assistant_response(
            "Here are some Python tutorials",
            turn,
            metadata={
                "confidence": 0.95,
                "sources_count": 5
            }
        )
        
        output = memory.call()
        
        # Check metadata is rendered
        assert "source: web_interface" in output
        assert "priority: high" in output
        assert "timestamp: 2024-01-01T10:00:00" in output
        assert "confidence: 0.95" in output
        assert "sources_count: 5" in output
    
    def test_metadata_filtering(self):
        """Test metadata filtering in template rendering.
        
        Tests:
        - Only specified metadata keys are shown
        - Other metadata is filtered out
        - Filtering works across multiple messages
        """
        memory = FlexibleConversationMemory()
        
        turn = memory.create_turn()
        memory.add_user_query(
            "Query with metadata",
            turn,
            metadata={
                "public": "visible",
                "private": "hidden",
                "internal": "secret"
            }
        )
        memory.add_assistant_response(
            "Response with metadata",
            turn,
            metadata={
                "public": "also visible",
                "confidence": "high",
                "debug": "hidden info"
            }
        )
        
        # Call with metadata filter
        output = memory.call(metadata_filter=["public", "confidence"])
        
        # Check filtered metadata
        assert "public: visible" in output
        assert "public: also visible" in output
        assert "confidence: high" in output
        
        # Check filtered out metadata is not present
        assert "private: hidden" not in output
        assert "internal: secret" not in output
        assert "debug: hidden info" not in output
    
    def test_system_messages_rendering(self):
        """Test system message rendering in the template.
        
        Tests:
        - System messages are properly formatted
        - System messages appear with "System:" prefix
        - Mixed message types work correctly
        """
        memory = FlexibleConversationMemory()
        
        turn = memory.create_turn()
        memory.add_system_message("You are a helpful assistant.", turn)
        memory.add_user_query("Hello", turn)
        memory.add_assistant_response("Hi! How can I help you?", turn)
        
        output = memory.call()
        
        # Check system message formatting
        assert "System: You are a helpful assistant." in output
        assert "User: Hello" in output
        assert "Assistant: Hi! How can I help you?" in output
        
        # Check order
        system_idx = output.index("System:")
        user_idx = output.index("User:")
        assistant_idx = output.index("Assistant:")
        assert system_idx < user_idx < assistant_idx
    
    def test_multiple_messages_same_role_in_turn(self):
        """Test multiple messages from the same role in one turn.
        
        Tests:
        - Multiple user queries in one turn
        - Multiple assistant responses in one turn
        - Proper ordering and formatting
        """
        memory = FlexibleConversationMemory()
        
        turn = memory.create_turn()
        memory.add_user_query("First question", turn)
        memory.add_user_query("Follow-up question", turn)
        memory.add_user_query("Another clarification", turn)
        memory.add_assistant_response("Comprehensive answer addressing all questions", turn)
        
        output = memory.call()
        
        # All messages should be present
        assert "User: First question" in output
        assert "User: Follow-up question" in output
        assert "User: Another clarification" in output
        assert "Assistant: Comprehensive answer" in output
        
        # Check ordering
        first_idx = output.index("First question")
        followup_idx = output.index("Follow-up question")
        clarification_idx = output.index("Another clarification")
        answer_idx = output.index("Comprehensive answer")
        
        assert first_idx < followup_idx < clarification_idx < answer_idx
    
    def test_empty_conversation_rendering(self):
        """Test rendering an empty conversation.
        
        Tests:
        - Empty conversation returns empty string
        - No errors are raised
        """
        memory = FlexibleConversationMemory()
        
        output = memory.call()
        assert output == ""
        
        # Also test with metadata filter
        output_filtered = memory.call(metadata_filter=["some_key"])
        assert output_filtered == ""
    
    def test_special_characters_in_content(self):
        """Test rendering with special characters in content.
        
        Tests:
        - Newlines in content
        - Special characters (quotes, brackets, etc.)
        - Unicode characters
        """
        memory = FlexibleConversationMemory()
        
        turn = memory.create_turn()
        memory.add_user_query(
            "Can you explain:\n1. Lists\n2. Dictionaries\n3. Sets",
            turn
        )
        memory.add_assistant_response(
            'Sure! Here\'s an explanation:\n• Lists: [1, 2, 3]\n• Dicts: {"key": "value"}\n• Sets: {1, 2, 3}',
            turn
        )
        
        output = memory.call()
        
        # Check special characters are preserved
        assert "1. Lists" in output
        assert "2. Dictionaries" in output
        assert '{"key": "value"}' in output
        assert "• Lists:" in output
    
    def test_complex_metadata_in_template(self):
        """Test rendering complex metadata structures.
        
        Tests:
        - Nested dictionaries in metadata
        - Lists in metadata
        - Numbers and booleans in metadata
        """
        memory = FlexibleConversationMemory()
        
        turn = memory.create_turn()
        memory.add_user_query(
            "Complex query",
            turn,
            metadata={
                "nested": {
                    "level1": {
                        "level2": "deep value"
                    }
                },
                "list_data": ["item1", "item2", "item3"],
                "numeric": 42,
                "boolean": True
            }
        )
        
        output = memory.call()
        
        # Check complex metadata is rendered (as string representations)
        assert "Complex query" in output
        # The template will render these as strings
        assert "nested:" in output
        assert "list_data:" in output
        assert "numeric: 42" in output
        assert "boolean: True" in output
    
    def test_template_with_none_metadata(self):
        """Test template handling of None metadata.
        
        Tests:
        - Messages with None metadata don't show metadata section
        - Messages with empty dict metadata don't show metadata section
        - Mixed messages with and without metadata
        """
        memory = FlexibleConversationMemory()
        
        turn = memory.create_turn()
        # Message with None metadata
        memory.add_user_query("No metadata query", turn, metadata=None)
        # Message with empty metadata
        memory.add_assistant_response("No metadata response", turn, metadata={})
        # Message with actual metadata
        memory.add_user_query("With metadata", turn, metadata={"key": "value"})
        
        output = memory.call()
        
        # Check messages are present
        assert "User: No metadata query" in output
        assert "Assistant: No metadata response" in output
        assert "User: With metadata" in output
        assert "key: value" in output
        
        # Count occurrences of "key:" - should only be one
        assert output.count("key:") == 1
    
    def test_callable_interface(self):
        """Test that the callable interface works the same as call().
        
        Tests:
        - __call__ produces same output as call()
        - Metadata filtering works with __call__
        """
        memory = FlexibleConversationMemory()
        
        turn = memory.create_turn()
        memory.add_user_query("Test query", turn, metadata={"meta1": "value1"})
        memory.add_assistant_response("Test response", turn, metadata={"meta2": "value2"})
        
        # Test call() and __call__() produce same output
        output_call = memory.call()
        output_callable = memory()
        assert output_call == output_callable
        
        # Test metadata filtering with both
        output_call_filtered = memory.call(metadata_filter=["meta1"])
        output_callable_filtered = memory(metadata_filter=["meta1"])
        assert output_call_filtered == output_callable_filtered
        assert "meta1: value1" in output_callable_filtered
        assert "meta2: value2" not in output_callable_filtered
    
    def test_template_direct_rendering(self):
        """Test the template can be rendered directly with Prompt.
        
        Tests:
        - CONVERSATION_TEMPLATE constant is valid
        - Can create Prompt with the template
        - Direct rendering matches memory.call()
        """
        from collections import OrderedDict
        
        # Create data structure manually
        turns = OrderedDict()
        turn_id = "test_turn"
        turns[turn_id] = [
            Message.from_user("Hello"),
            Message.from_assistant("Hi there!")
        ]
        
        # Render with Prompt directly
        prompt = Prompt(
            template=CONVERSATION_TEMPLATE,
            prompt_kwargs={
                "turns": turns,
                "metadata_filter": None
            }
        )
        direct_output = prompt.call().strip()
        
        # Create same conversation with memory
        memory = FlexibleConversationMemory()
        memory_turn = memory.create_turn()
        memory.add_user_query("Hello", memory_turn)
        memory.add_assistant_response("Hi there!", memory_turn)
        memory_output = memory.call().strip()
        
        # Should produce same output
        assert direct_output == memory_output
    
    def test_long_conversation_rendering(self):
        """Test rendering of long conversations.
        
        Tests:
        - Many turns (10+)
        - Many messages per turn
        - Performance doesn't degrade
        """
        memory = FlexibleConversationMemory()
        
        # Create 10 turns with multiple messages each
        for i in range(10):
            turn = memory.create_turn()
            memory.add_user_query(f"Question {i+1}", turn)
            memory.add_user_query(f"Clarification {i+1}", turn)
            memory.add_assistant_response(f"Answer {i+1}", turn)
            if i % 2 == 0:
                memory.add_system_message(f"System note {i+1}", turn)
        
        output = memory.call()
        
        # Check all messages are present
        for i in range(10):
            assert f"Question {i+1}" in output
            assert f"Clarification {i+1}" in output
            assert f"Answer {i+1}" in output
            if i % 2 == 0:
                assert f"System note {i+1}" in output
        
        # Check it's not empty and has reasonable length
        assert len(output) > 500  # Should be a long conversation
        
        # Check ordering is maintained
        q1_idx = output.index("Question 1")
        q10_idx = output.index("Question 10")
        assert q1_idx < q10_idx


class TestTemplateEdgeCases:
    """Test edge cases and error conditions in template rendering."""
    
    def test_template_with_jinja2_special_chars_in_content(self):
        """Test content that contains Jinja2 special characters.
        
        Tests:
        - Content with {{ }} doesn't break template
        - Content with {% %} doesn't break template
        - Content is properly escaped
        """
        memory = FlexibleConversationMemory()
        
        turn = memory.create_turn()
        memory.add_user_query(
            "Show me a template: {{ variable }} and {% if condition %}",
            turn
        )
        memory.add_assistant_response(
            "Here's the template: {{ var }} and {% for item in items %}",
            turn
        )
        
        output = memory.call()
        
        # Special characters should be preserved in output
        assert "{{ variable }}" in output
        assert "{% if condition %}" in output
        assert "{{ var }}" in output
        assert "{% for item in items %}" in output
    
    def test_metadata_with_jinja2_chars(self):
        """Test metadata containing Jinja2 special characters.
        
        Tests:
        - Metadata with template syntax doesn't break rendering
        - Values are properly escaped
        """
        memory = FlexibleConversationMemory()
        
        turn = memory.create_turn()
        memory.add_user_query(
            "Query",
            turn,
            metadata={
                "template": "{{ user_input }}",
                "condition": "{% if x > 0 %}",
                "normal": "regular value"
            }
        )
        
        output = memory.call()
        
        # Metadata should be rendered safely
        assert "template: {{ user_input }}" in output
        assert "condition: {% if x > 0 %}" in output
        assert "normal: regular value" in output
    
    def test_very_long_messages(self):
        """Test rendering very long messages.
        
        Tests:
        - Long messages don't break template
        - Full content is preserved
        """
        memory = FlexibleConversationMemory()
        
        # Create a very long message
        long_content = "This is a very long message. " * 100  # ~2500 characters
        
        turn = memory.create_turn()
        memory.add_user_query(long_content, turn)
        memory.add_assistant_response("Short response", turn)
        
        output = memory.call()
        
        # Check long content is fully preserved
        assert long_content in output
        assert "Short response" in output
    
    def test_whitespace_preservation(self):
        """Test that whitespace in messages is preserved.
        
        Tests:
        - Leading/trailing spaces
        - Multiple spaces
        - Tabs and newlines
        """
        memory = FlexibleConversationMemory()
        
        turn = memory.create_turn()
        memory.add_user_query("  Message with   spaces  ", turn)
        memory.add_assistant_response("\tTabbed\tmessage\t", turn)
        memory.add_system_message("Line1\nLine2\nLine3", turn)
        
        output = memory.call()
        
        # Check whitespace is preserved
        assert "  Message with   spaces  " in output
        assert "\tTabbed\tmessage\t" in output
        assert "Line1\nLine2\nLine3" in output