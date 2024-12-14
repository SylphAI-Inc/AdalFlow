"""Memory component for user-assistant conversations.

Memory can include data modeling, in-memory data storage, local file data storage, cloud data persistence, data pipeline, data retriever.
It is itself an LLM application and different use cases can do it differently.

This component handles the storage and retrieval of conversation history between users
and assistants. It provides local memory experience with the ability to format and
return conversation history.

Attributes:
    current_conversation (Conversation): Stores the current active conversation.
    turn_db (LocalDB): Database for storing all conversation turns.
    conver_db (LocalDB): Database for storing complete conversations.
"""

from uuid import uuid4
from adalflow.core.component import Component
from adalflow.core.db import LocalDB
from adalflow.core.types import (
    Conversation,
    DialogTurn,
    UserQuery,
    AssistantResponse,
)


class Memory(Component):
    def __init__(self, turn_db: LocalDB = None):
        """Initialize the Memory component.

        Args:
            turn_db (LocalDB, optional): Database for storing conversation turns.
                Defaults to None, in which case a new LocalDB is created.
        """
        super().__init__()
        self.current_conversation = Conversation()
        self.turn_db = turn_db or LocalDB()  # all turns
        self.conver_db = LocalDB()  # a list of conversations

    def call(self) -> str:
        """Returns the current conversation history as a formatted string.

        Returns:
            str: Formatted conversation history with alternating user and assistant messages.
                Returns empty string if no conversation history exists.
        """
        if not self.current_conversation.dialog_turns:
            return ""

        formatted_history = []
        for turn in self.current_conversation.dialog_turns.values():
            formatted_history.extend(
                [
                    f"User: {turn.user_query.query_str}",
                    f"Assistant: {turn.assistant_response.response_str}",
                ]
            )
        return "\n".join(formatted_history)

    def add_dialog_turn(self, user_query: str, assistant_response: str):
        """Add a new dialog turn to the current conversation.

        Args:
            user_query (str): The user's input message.
            assistant_response (str): The assistant's response message.
        """
        dialog_turn = DialogTurn(
            id=str(uuid4()),
            user_query=UserQuery(query_str=user_query),
            assistant_response=AssistantResponse(response_str=assistant_response),
        )

        self.current_conversation.append_dialog_turn(dialog_turn)

        self.turn_db.add(
            {"user_query": user_query, "assistant_response": assistant_response}
        )
