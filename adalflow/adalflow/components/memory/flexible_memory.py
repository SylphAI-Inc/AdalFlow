"""Flexible conversation memory with turns containing multiple messages.

This memory design uses an OrderedDict where each turn_id maps to a list of messages.
This allows multiple user queries and assistant responses within the same turn.
"""

from uuid import uuid4
from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime
from collections import OrderedDict
from adalflow.core.component import Component
from adalflow.core.db import LocalDB
from adalflow.core.types import DataClass
from adalflow.core.prompt_builder import Prompt


@dataclass
class Message(DataClass):
    """A single message in a conversation."""
    id: str = field(default_factory=lambda: str(uuid4()))
    role: Literal["user", "assistant", "system"] = "user"
    content: str = ""
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def from_user(cls, content: str, metadata: Optional[Dict] = None):
        """Create a user message."""
        return cls(role="user", content=content, metadata=metadata)
    
    @classmethod
    def from_assistant(cls, content: str, metadata: Optional[Dict] = None):
        """Create an assistant message."""
        return cls(role="assistant", content=content, metadata=metadata)
    
    @classmethod
    def from_system(cls, content: str, metadata: Optional[Dict] = None):
        """Create a system message."""
        return cls(role="system", content=content, metadata=metadata)


@dataclass
class Conversation(DataClass):
    """A conversation organized as turns, where each turn can have multiple messages."""
    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    turns: OrderedDict = field(default_factory=OrderedDict)  # turn_id -> List[Message]
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    _current_turn_id: Optional[str] = field(default=None, init=False)
    
    def add_message_to_turn(self, turn_id: str, message: Message) -> str:
        """Add a message to a specific turn.
        
        Args:
            turn_id: The turn identifier
            message: The message to add
            
        Returns:
            str: The message ID
        """
        if turn_id not in self.turns:
            self.turns[turn_id] = []
        self.turns[turn_id].append(message)
        return message.id
    
    def get_turn_messages(self, turn_id: str) -> List[Message]:
        """Get all messages in a specific turn."""
        return self.turns.get(turn_id, [])
    
    def get_all_messages(self) -> List[Message]:
        """Get all messages in order across all turns."""
        messages = []
        for turn_messages in self.turns.values():
            messages.extend(turn_messages)
        return messages
    
    def get_messages_by_role(self, role: str) -> List[Message]:
        """Get all messages from a specific role."""
        messages = []
        for turn_messages in self.turns.values():
            messages.extend([msg for msg in turn_messages if msg.role == role])
        return messages
    
    def get_last_user_message(self) -> Optional[Message]:
        """Get the most recent user message."""
        for turn_messages in reversed(list(self.turns.values())):
            for msg in reversed(turn_messages):
                if msg.role == "user":
                    return msg
        return None
    
    def get_last_assistant_message(self) -> Optional[Message]:
        """Get the most recent assistant message."""
        for turn_messages in reversed(list(self.turns.values())):
            for msg in reversed(turn_messages):
                if msg.role == "assistant":
                    return msg
        return None
    
    def create_turn(self) -> str:
        """Create a new turn and return its ID."""
        turn_id = str(uuid4())
        self.turns[turn_id] = []
        self._current_turn_id = turn_id
        return turn_id


# Template for conversation formatting
CONVERSATION_TEMPLATE = r"""
{% for turn_id, messages in turns.items() -%}
{% for message in messages -%}
{% if message.role == "user" -%}
User: {{ message.content }}
{% if message.metadata -%}
{% for key, value in message.metadata.items() -%}
{% if not metadata_filter or key in metadata_filter -%}
{{ key }}: {{ value }}
{% endif -%}
{% endfor -%}
{% endif -%}
{% elif message.role == "assistant" -%}
Assistant: {{ message.content }}
{% if message.metadata -%}
{% for key, value in message.metadata.items() -%}
{% if not metadata_filter or key in metadata_filter -%}
{{ key }}: {{ value }}
{% endif -%}
{% endfor -%}
{% endif -%}
{% elif message.role == "system" -%}
System: {{ message.content }}
{% endif -%}
{% endfor -%}
{% endfor -%}"""


class FlexibleConversationMemory(Component):
    """A flexible conversation memory with turns containing multiple messages."""
    
    def __init__(self, turn_db: LocalDB = None, user_id: str = None):
        """Initialize the flexible memory component.
        
        Args:
            turn_db: Database for storing messages
            user_id: Optional user identifier
        """
        super().__init__()
        self.current_conversation = Conversation(user_id=user_id)
        self.message_db = turn_db or LocalDB()  # Store all messages
        self.conver_db = LocalDB()  # Store complete conversations
        self.user_id = user_id
        
    def clear_conversation(self):
        """Clear all turns and messages in the current conversation."""
        self.current_conversation.turns.clear()
        self.current_conversation._current_turn_id = None
        
    def clear_conversation_turns(self):
        """Alias for clear_conversation for compatibility."""
        self.clear_conversation()
        
    def new_conversation(self):
        """Start a new conversation, saving the current one."""
        # Save current conversation if it has messages
        if self.current_conversation.turns:
            self.conver_db.add(self.current_conversation)
        
        # Create new conversation
        self.current_conversation = Conversation(user_id=self.user_id)
        
    def create_turn(self) -> str:
        """Create a new turn and return its ID.
        
        Returns:
            str: The new turn ID
        """
        return self.current_conversation.create_turn()
        
    def add_user_query(self, content: str, metadata: Optional[Dict] = None, turn_id: Optional[str] = None) -> str:
        """Add a user message to a turn.
        
        Args:
            content: The user's message content
            metadata: Optional metadata
            turn_id: Optional turn ID. If None, creates a new turn.
            
        Returns:
            str: The turn ID the message was added to
        """
        # Use provided turn_id or create new turn
        if turn_id is None:
            turn_id = self.create_turn()
        elif turn_id not in self.current_conversation.turns:
            # Turn doesn't exist, create it
            self.current_conversation.turns[turn_id] = []
            
        # Track as current turn
        self.current_conversation._current_turn_id = turn_id
        
        # Create and add the user message
        message = Message.from_user(content, metadata)
        self.current_conversation.add_message_to_turn(turn_id, message)
        
        # Store in database
        self.message_db.add({
            "message_id": message.id,
            "turn_id": turn_id,
            "role": "user",
            "content": content,
            "metadata": metadata,
            "timestamp": message.timestamp
        })
        
        return turn_id
    
    def add_assistant_response(
        self, 
        content: str,
        metadata: Optional[Dict] = None,
        turn_id: Optional[str] = None
    ) -> str:
        """Add an assistant message to a turn.
        
        Args:
            content: The assistant's message content
            metadata: Optional metadata
            turn_id: Optional turn ID. If None, uses current turn or creates new.
                    
        Returns:
            str: The turn ID the message was added to
        """
        # Determine which turn to use
        if turn_id is None:
            if self.current_conversation._current_turn_id:
                turn_id = self.current_conversation._current_turn_id
            else:
                # No active turn, create new one for standalone response
                turn_id = self.create_turn()
        elif turn_id not in self.current_conversation.turns:
            # Turn doesn't exist, create it
            self.current_conversation.turns[turn_id] = []
        
        # Create and add the assistant message
        message = Message.from_assistant(content, metadata)
        self.current_conversation.add_message_to_turn(turn_id, message)
        
        # Store in database
        self.message_db.add({
            "message_id": message.id,
            "turn_id": turn_id,
            "role": "assistant",
            "content": content,
            "metadata": metadata,
            "timestamp": message.timestamp
        })
        
        return turn_id
    
    def add_system_message(self, content: str, metadata: Optional[Dict] = None, turn_id: Optional[str] = None) -> str:
        """Add a system message to a turn.
        
        Args:
            content: The system message content
            metadata: Optional metadata
            turn_id: Optional turn ID. If None, creates a new turn.
            
        Returns:
            str: The turn ID the message was added to
        """
        # Use provided turn_id or create new turn
        if turn_id is None:
            turn_id = self.create_turn()
        elif turn_id not in self.current_conversation.turns:
            self.current_conversation.turns[turn_id] = []
            
        # Create and add the system message
        message = Message.from_system(content, metadata)
        self.current_conversation.add_message_to_turn(turn_id, message)
        
        # Store in database
        self.message_db.add({
            "message_id": message.id,
            "turn_id": turn_id,
            "role": "system",
            "content": content,
            "metadata": metadata,
            "timestamp": message.timestamp
        })
        
        return turn_id
    
    def get_turn_messages(self, turn_id: str) -> List[Message]:
        """Get all messages for a specific turn.
        
        Args:
            turn_id: The turn identifier
            
        Returns:
            List of messages in that turn
        """
        return self.current_conversation.get_turn_messages(turn_id)
    
    def get_current_turn_id(self) -> Optional[str]:
        """Get the current turn ID if any."""
        return self.current_conversation._current_turn_id
    
    def call(self, metadata_filter: Optional[List[str]] = None) -> str:
        """Get the conversation history as a formatted string.
        
        Args:
            metadata_filter: Optional list of metadata keys to include
            
        Returns:
            str: Formatted conversation history
        """
        if not self.current_conversation.turns:
            return ""
            
        prompt = Prompt(
            template=CONVERSATION_TEMPLATE,
            prompt_kwargs={
                "turns": self.current_conversation.turns,
                "metadata_filter": metadata_filter,
            },
        )
        return prompt.call().strip()
    
    def get_all_messages(self) -> List[Message]:
        """Get all messages in order across all turns."""
        return self.current_conversation.get_all_messages()
    
    def get_last_n_messages(self, n: int) -> List[Message]:
        """Get the last n messages across all turns."""
        messages = self.get_all_messages()
        return messages[-n:] if len(messages) >= n else messages
    
    def count_messages(self) -> Dict[str, int]:
        """Count messages by role."""
        counts = {"user": 0, "assistant": 0, "system": 0}
        for msg in self.get_all_messages():
            counts[msg.role] = counts.get(msg.role, 0) + 1
        return counts
    
    def count_turns(self) -> int:
        """Count the number of turns."""
        return len(self.current_conversation.turns)
    
    def reset_pending_query(self):
        """Reset current turn tracking. Included for compatibility."""
        # Don't clear the current turn, just unset it as "current"
        # This allows adding more messages to existing turns if needed
        self.current_conversation._current_turn_id = None
        
    def __call__(self, metadata_filter: Optional[List[str]] = None) -> str:
        """Make the memory callable to get conversation history."""
        return self.call(metadata_filter)
        
    def __len__(self) -> int:
        """Return the number of messages."""
        return len(self.get_all_messages())