import pytest
from uuid import uuid4
from core.data_classes import UserQuery, AssistantResponse, DialogTurn, DialogSession


def test_user_query_creation():
    query = "What is the weather today?"
    metadata = {"location": "New York"}
    user_query = UserQuery(query_str=query, metadata=metadata)
    assert user_query.query_str == query
    assert user_query.metadata == metadata


def test_assistant_response_creation():
    response = "It's sunny in New York today."
    metadata = {"temperature": "75°F"}
    assistant_response = AssistantResponse(response_str=response, metadata=metadata)
    assert assistant_response.response_str == response
    assert assistant_response.metadata == metadata


def test_dialog_turn_creation():
    user_query = UserQuery(query_str="Hello, how are you?")
    assistant_response = AssistantResponse(response_str="I'm fine, thank you!")
    dialog_turn = DialogTurn(
        id=str(uuid4()), user_query=user_query, assistant_response=assistant_response
    )
    assert dialog_turn.user_query == user_query
    assert dialog_turn.assistant_response == assistant_response
    assert isinstance(dialog_turn.id, str)


def test_dialog_session_operations():
    session = DialogSession()
    assert isinstance(session.id, str)  # Check if the UUID is automatically generated

    # Creating dialog turns
    user_query = UserQuery(query_str="Hello, how are you?")
    assistant_response = AssistantResponse(response_str="I'm fine, thank you!")
    dialog_turn = DialogTurn(
        id=str(uuid4()), user_query=user_query, assistant_response=assistant_response
    )

    session.append_dialog_turn(dialog_turn)
    assert len(session.dialog_turns) == 1
    assert (
        session.dialog_turns[0] == dialog_turn
    ), f"Expected {session.dialog_turns} in session."

    # Testing order enforcement
    with pytest.raises(AssertionError):
        wrong_turn = DialogTurn(
            id=str(uuid4()),
            order=2,
            user_query=user_query,
            assistant_response=assistant_response,
        )
        session.append_dialog_turn(wrong_turn)

    # Update and delete operations
    new_response = AssistantResponse(response_str="Actually, I'm great!")
    updated_turn = DialogTurn(
        id=dialog_turn.id, user_query=user_query, assistant_response=new_response
    )
    session.update_dialog_turn(1, updated_turn)
    assert (
        session.dialog_turns[1].assistant_response.response_str
        == "Actually, I'm great!"
    )

    session.delete_dialog_turn(1)
    assert len(session.dialog_turns) == 1
