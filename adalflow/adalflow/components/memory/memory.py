"""Simple converation memory component for user-assistant conversations.

Attributes:
    current_conversation (Conversation): Stores the current active conversation.
    turn_db (LocalDB): Database for storing all conversation turns.
    conver_db (LocalDB): Database for storing complete conversations.
"""

from uuid import uuid4
from typing import Union, Optional, List
from adalflow.core.component import Component
from adalflow.core.db import LocalDB
from adalflow.core.types import (
    Conversation,
    DialogTurn,
    UserQuery,
    AssistantResponse,
)
from adalflow.core.prompt_builder import Prompt


# Jinja2 template for formatting conversation history with dialog turns being ordered
# dialog_turns is an OrderedDict[int, DialogTurn]
CONVERSATION_TEMPLATE = r"""
{% for order, turn in dialog_turns.items() -%}
{% if turn.user_query -%}
User:
query: {{ turn.user_query.query_str }}
{% if turn.user_query.metadata -%}
{% for key, value in turn.user_query.metadata.items() -%}
{% if not metadata_filter or key in metadata_filter -%}
{{ key }}: {{ value }}
{% endif -%}
{% endfor -%}
{% endif -%}
{% endif -%}
{% if turn.assistant_response -%}
Assistant: {{ turn.assistant_response.response_str }}
{% if turn.assistant_response.metadata -%}
{% for key, value in turn.assistant_response.metadata.items() -%}
{% if not metadata_filter or key in metadata_filter -%}
{{ key }}: {{ value }}
{% endif -%}
{% endfor -%}
{% endif -%}
{% endif -%}
{% if not loop.last %}
{% endif -%}
{% endfor -%}"""


class ConversationMemory(Component):
    def __init__(self, turn_db: LocalDB = None, user_id: str = None):
        """Initialize the Memory component.

        Args:
            turn_db (LocalDB, optional): Database for storing conversation turns.
                Defaults to None, in which case a new LocalDB is created.
        """
        super().__init__()
        self.current_conversation = Conversation(user_id=user_id)
        self.turn_db = turn_db or LocalDB()  # all turns
        self.conver_db = LocalDB()  # a list of conversations
        self._pending_user_query = None  # Store pending user query
        self.user_id = user_id

    def clear_conversation_turns(self):
        self._pending_user_query = None
        self.current_conversation.dialog_turns.clear()

    def new_conversation(self):
        # save the current conversation to the conversation database
        self.conver_db.add(self.current_conversation)

        self.current_conversation = Conversation(user_id=self.user_id)

    def call(self, metadata_filter: Optional[List[str]] = None) -> str:
        """Returns the current conversation history as a formatted string.

        Args:
            metadata_filter (Optional[List[str]]): List of metadata keys to include.
                If None, all metadata is included.

        Returns:
            str: Formatted conversation history with alternating user and assistant messages.
                Returns empty string if no conversation history exists.
        """
        if not self.current_conversation.dialog_turns:
            return ""

        prompt = Prompt(
            template=CONVERSATION_TEMPLATE,
            prompt_kwargs={
                "dialog_turns": self.current_conversation.dialog_turns,
                "metadata_filter": metadata_filter,
            },
        )
        return prompt.call().strip()

    # Note: not used
    def add_dialog_turn(
        self,
        user_query: Union[str, UserQuery],
        assistant_response: Union[str, AssistantResponse],
    ):
        """Add a new dialog turn to the current conversation.

        Args:
            user_query (str): The user's input message.
            assistant_response (str): The assistant's response message.
        """
        user_query = (
            user_query
            if isinstance(user_query, UserQuery)
            else UserQuery(query_str=user_query)
        )
        assistant_response = (
            assistant_response
            if isinstance(assistant_response, AssistantResponse)
            else AssistantResponse(response_str=assistant_response)
        )

        dialog_turn = DialogTurn(
            id=str(uuid4()),
            user_query=user_query,
            assistant_response=assistant_response,
            conversation_id=self.current_conversation.id,
            # order will be automatically set by append_dialog_turn
        )

        self.current_conversation.append_dialog_turn(dialog_turn)

        self.turn_db.add(
            {"user_query": user_query, "assistant_response": assistant_response}
        )

    def add_user_query(self, user_query: Union[str, UserQuery]) -> str:
        """Add a user query to start a new dialog turn.

        Args:
            user_query (Union[str, UserQuery]): The user's input message.
                If UserQuery object, can include metadata.

        Returns:
            str: The ID of the pending dialog turn.

        Raises:
            ValueError: If there's already a pending user query without an assistant response.
        """
        if self._pending_user_query is not None:
            raise ValueError(
                "There's already a pending user query. Please add an assistant response first."
            )

        user_query = (
            user_query
            if isinstance(user_query, UserQuery)
            else UserQuery(query_str=user_query)
        )

        # Create a new dialog turn with just the user query
        turn_id = str(uuid4())
        self._pending_user_query = {
            "id": turn_id,
            "user_query": user_query,
            "order": self.current_conversation.get_next_order(),
        }

        return turn_id

    def add_assistant_response(
        self, assistant_response: Union[str, AssistantResponse]
    ) -> str:
        """Add an assistant response to complete the current dialog turn.

        Args:
            assistant_response (Union[str, AssistantResponse]): The assistant's response message.

        Returns:
            str: The ID of the completed dialog turn.

        Raises:
            ValueError: If there's no pending user query to respond to.
        """
        if self._pending_user_query is None:
            raise ValueError(
                "No pending user query found. Please add a user query first."
            )

        assistant_response = (
            assistant_response
            if isinstance(assistant_response, AssistantResponse)
            else AssistantResponse(response_str=assistant_response)
        )

        # Create and add the complete dialog turn
        dialog_turn = DialogTurn(
            id=self._pending_user_query["id"],
            conversation_id=self.current_conversation.id,
            user_query=self._pending_user_query["user_query"],
            assistant_response=assistant_response,
            order=self._pending_user_query["order"],
        )


        # current conversation manages the display to LLM.
        self.current_conversation.append_dialog_turn(dialog_turn)

        # Store in database
        self.turn_db.add(
            {
                "user_query": self._pending_user_query["user_query"],
                "assistant_response": assistant_response,
            }
        )

        # Clear the pending query
        turn_id = self._pending_user_query["id"]
        self._pending_user_query = None

        return turn_id

    def reset_pending_query(self):
        """Reset any pending query state. Useful for cleanup after cancellations or before starting new queries."""
        self._pending_user_query = None
