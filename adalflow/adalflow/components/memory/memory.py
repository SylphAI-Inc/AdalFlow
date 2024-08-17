"""Memory for user-assistant conversations. [Not completed]

Memory can include data modeling, in-memory data storage, local file data storage, cloud data persistence, data pipeline, data retriever.
It is itself an LLM application and different use cases can do it differently.


This implementation covers the minimal and local memory experience for the user-assistant conversation.
"""

from adalflow.core.types import (
    Conversation,
)

from adalflow.core.db import LocalDB
from adalflow.core.component import Component


class Memory(Component):
    def __init__(self, turn_db: LocalDB = None):
        super().__init__()
        self.current_convesation = Conversation()
        self.turn_db = turn_db or LocalDB()  # all turns
        self.conver_db = LocalDB()  # a list of conversations
