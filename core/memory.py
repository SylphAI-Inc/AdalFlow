"""
Memory is more of a db where you can save all users' data and retrieve it when needed.

We can control if the memory is just per-session or retrieve from the users' all history.

The main form of the memory is a list of DialogSessions, where each DialogSession is a list of DialogTurns.
When memory becomes too large, we need to (1) compress (2) RAG to retrieve the most relevant memory.

In this case, we only manage the memory for the current session.
"""

from core.component import Component
