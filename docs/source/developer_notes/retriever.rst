Retriever
============

**Why Retriever for LLM?**

LLMs halluciate and also has knowledge cut-off. External and relevant context is needed to increase the factuality and relevancy on the LLM answers.
Due to LLM's context window limit(can only take so much tokens each time), the ``lost-in-the-middle`` problem[6], and the high cost on speed and resources using large context,
it is practical to use a retriever to retrieve the most relevant information to get the best performance. Retrieval Augemented Generation (RAG)[7] applications become one of main applications in LLMs.

**What is a retriever?**

Though the definition is simple - "Retrieve relevant information for a given query from a given database", retriever is the part with the most diversity in LLM application landscapes.
It is essentially the core of any search and information retrieval system. 

There are numerous search techniques, such as keyword search, full-text search, semantic search, and reranking, applied on various data types, such as text, time-sensitive data, locations, sensor data, and images, videos, audios, etc, stored in various types of databases, such as relational databases, NoSQL databases, and vector databases.
There are also dense and sparse retrieval methods.
.. - Keyword search
.. - Full-text search: Here is one example: https://www.postgresql.org/docs/current/textsearch.html
..   > TF-IDF (Term Frequency-Inverse Document Frequency)
..   > BM25 (Best Matching 25)
.. - Wildcard search, Fuzzy search, Proximity search, Phrase search, Boolean search, facet search etc
.. - Semantic search using embedding models 
.. - Reranking using ranking models.

.. Second, there are numerous data types: Text, Time-sensitive data, Locations, Sensor data, and Images, Videos, Audios etc

.. Third,  the data can be stored anywhere: In-memory data, Local and Disk-based data, and Cloud DBs such as relational databases, NoSQL databases, vector databases etc

**What is important for users?**

The most important thing is for users to design the search strategy that can be a combination of all different search techniques. 
For example, when you want to search for candidates from a pool of profiles stored in realtional db say Postgres, you can search by name simply using keyword, or check if the name equals to the query.
You also want to search by their profession, which you have already categorized, this makes it a filter search. Or the search query semantically is hard to describe with keywords or to create a fixed set of category, then vector/semantc search using 
the cosine similarity between the query and the profile embeddings can be helpful. To be even more accurate, reranking models where the query and a candidate set of the profile text are directly passed to models and be reranked.
So in production, retrieval often is multiple-stages, and each stage can use different types of search techniques to filter out the candidates, from the cheapest to the most expensive and most accurate.
As a library, we do not aim to optimize the coverage of integration, but provide a design pattern so that:

**What is retriever in LightRAG library?**

A retriever in our library is a component that potentially retrieves relevant ``context`` and pass it to the ``prompt`` of a ``generator``.
If your data is big, we assume it is users' responsibility to do fuzzy and cheap filter and search that gives high recall even though low precision till to have a manageable set of candidates (fit into local memory or a latency limit) to optimize for high precision. 
To optimize recall, often BM25, TF-IDF, and semantic search using embedding models are used. And lastly, reranking models are used for the final precision optimization.
As the layer close to deliver the final user experience, we try to provide a great design pattern so that:

- Users can clearly implement your own retriever to work with your data and your LightRAG LLM applications.
- Know how to evaluate and optimize the task pipeline.


Design pattern
------------------
A retrieval will work hand in hand with a ``database``: the retriever will be responsible for building and querying the index and work with a database, either local or cloud to save and load index.

A retriever will retrieve the `ids` of the ``top_k`` most relevant documents given a query. The user can then use these `ids` to retrieve the actual documents from the database.

RetrieverOutput
^^^^^^^^^^^^^^^^^^^^^^^^
Thus our ``RetrieverOutput`` is defined as:

.. code-block:: python

    class RetrieverOutput(DataClass):
        r"""Mainly used to retrieve a list of documents with scores."""

        doc_indexes: List[int]  # either index or ids potentially
        doc_scores: Optional[List[float]] = None
        query: Optional[str] = None
        documents: Optional[List[Document]] = None  # TODO: documents can be of any time





Build and Query Index -- The Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For all retrievers, they will compute/manage ``index`` and handles the  ``query`` on given database.
Index is data-structure specific to either retrieval method that is used to compute a relevancy score in the case of embeddings for semantic search and Term-Frequency-Inverse Document Frequency (TF-IDF) for BM25, and for rerankers it is just the query and the candidates files themselves and the model.
For a local retriever, it will need to (1) computes the index itself given candidates documents, persist them for later usage (2) load index from local or cloud storage (3) query the index to get the relevant documents.

Load and Save Index - The Data Storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For loading and saving in local and disk storage, we opt for ``pickle``, additionally, you can use local database such as SQLite, PgVector, Postgres along with cloud version to persist the index.


Current Coverage 
--------------------

To implement three local retrievers to work on local documents and data types to showcase these algorithms:

1. ``BM25Retriever`` 
2. ``FAISSRetriever`` using FAISS library for semantic search
3. ``Reranker`` a local reranker model.

To demonstrate how we can use search provided by cloud database, we can consider them as a search service providers:

1. ``PostgresRetriever`` for full-text search together with either ``SQLAlchemy`` or ``Psycopg2``
2. ``PineConeRetriever`` for semantic search using PineCone API.

Remeber: they are the service proviers and the evaluation lies in developers hands and can be unique to your data and applications.

Examples 
------------------

Local FAISSRetriever
^^^^^^^^^^^^^^^^^^^^^^^^

Local BM25Retriever
^^^^^^^^^^^^^^^^^^^^^^^^

PostgresRetriever
^^^^^^^^^^^^^^^^^^^^^^^^

PineConeRetriever
^^^^^^^^^^^^^^^^^^^^^^^^

CohereReRanker
^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: References
   :class: highlight

   1. Full-text search on PostgreSQL: https://www.postgresql.org/docs/current/textsearch.html
   2. BM25: https://en.wikipedia.org/wiki/Okapi_BM25
   3. Representative learning models: https://arxiv.org/abs/2104.08663 [Find the right reference]
   4. Reranking models: https://arxiv.org/abs/2104.08663 [Find the right reference]
   5. FAISS: 
   6. Lost-in-the-middle: https://arxiv.org/abs/2104.08663 [Find the right reference]
   7. RAG: https://arxiv.org/abs/2104.08663 [Find the first paper on RAG]