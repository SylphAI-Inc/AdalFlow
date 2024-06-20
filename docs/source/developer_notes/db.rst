Database
====================

So far, we have seen how our core components like ``Generator``, ``Embedder``, and ``Retriever`` work without any database and enforced data format to read data from and to write data to.
However, in real-world LLM applications, we can not avoid to deal with data storage:

1. Our documents to retrieve context from can be large and be stored in a file system or in a database in forms of tables or graphs.
2. We often need to pre-process a large amount of data (like text splitting and embedding and idf in BM25) in a datapipline into a cloud database.
3. We need to write records, logs to files or databases for monitoring and debugging.
4. When it comes to applications where states matter, like games and chatbots, we need to store the states and conversational history.

We created :class:`core.types.Document` and :class:`core.types.DialogTurn` to help with text document and conversational histor processing and data storage.

Data Models
--------------------

Local database
--------------------

Cloud database
--------------------

Graph database
--------------------
