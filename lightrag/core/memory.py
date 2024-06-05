"""
Memory is more of a db where you can save all users' data and retrieve it when needed.

We can control if the memory is just per-session or retrieve from the users' all history.

The main form of the memory is a list of DialogSessions, where each DialogSession is a list of DialogTurns.
When memory becomes too large, we need to (1) compress (2) RAG to retrieve the most relevant memory.

In this case, we only manage the memory for the current session.
"""

from lightrag.core.component import Component
from lightrag.core.data_classes import (
    DialogSession,
    DialogTurn,
    UserQuery,
    AssistantResponse,
)


class Memory(Component):
    def __init__(self):
        super().__init__()
        self.memory = DialogSession()

    def __call__(self) -> str:
        return self.memory.get_chat_history_str()

    def add_dialog_turn(self, user_query: str, assistant_response: str):
        self.memory.append_dialog_turn(
            DialogTurn(
                user_query=UserQuery(user_query),
                assistant_response=AssistantResponse(assistant_response),
            )
        )
