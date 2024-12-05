from adalflow.components.memory.memory import Memory


def test_empty_memory():
    memory = Memory()
    assert memory() == ""


def test_add_dialog_turn():
    memory = Memory()
    memory.add_dialog_turn("Hello", "Hi! How can I help you?")
    expected = "User: Hello\nAssistant: Hi! How can I help you?"
    assert memory() == expected


def test_multiple_turns():
    memory = Memory()
    memory.add_dialog_turn("Hello", "Hi!")
    memory.add_dialog_turn("How are you?", "I'm good!")
    expected = (
        "User: Hello\n" "Assistant: Hi!\n" "User: How are you?\n" "Assistant: I'm good!"
    )
    assert memory() == expected
