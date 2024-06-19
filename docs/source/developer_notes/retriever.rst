Retriever
============

Context
------------------

**Why Retriever for LLM?**

LLMs halluciate and also has knowledge cut-off. External and relevant context is needed to increase the factuality, relevancy, and freshness on the LLM answers.
Due to LLM's context window limit(can only take so much tokens each time), the ``lost-in-the-middle`` problem[6], and the high cost on speed and resources using large context,
it is practical to use a retriever to retrieve the most relevant information to get the best performance. Retrieval Augemented Generation (RAG)[7] applications become one of main applications in LLMs.

.. It is easy to build a demo, but hard to excel due to the many different parts in the pipeline that made it difficult to evaluate and to optimize.

**What is a retriever?**

Though the definition is simple - "Retrieve relevant information for a given query from a given database", retriever can go as wide as the entire search and information retriever field.
No doubt the retriever part is the one of the most diversity and have the longest history in LLM application landscapes.


There are numerous search techniques long existing before the vector/semantic search and a reranking model, such as keyword search, fuzzy search, proximity search, phrase search, boolean search, facet search, full-text search,
and can be applied on various data types, such as text, time-sensitive data, locations, sensor data, and images, videos, audios, etc, and stored in various types of databases, such as relational databases, NoSQL databases, and vector databases.

.. In LightRAG
.. There are also dense and sparse retrieval methods.
.. - Keyword search
.. - Full-text search: Here is one example: https://www.postgresql.org/docs/current/textsearch.html
..   > TF-IDF (Term Frequency-Inverse Document Frequency)
..   > BM25 (Best Matching 25)
.. - Wildcard search, Fuzzy search, Proximity search, Phrase search, Boolean search, facet search etc
.. - Semantic search using embedding models
.. - Reranking using ranking models.

.. Second, there are numerous data types: Text, Time-sensitive data, Locations, Sensor data, and Images, Videos, Audios etc

.. Third,  the data can be stored anywhere: In-memory data, Local and Disk-based data, and Cloud DBs such as relational databases, NoSQL databases, vector databases etc

**Retrievel in production**

So in production, retrieval is often of multiple-stages, from the cheapest to the most expensive and most accurate, from millions of candidates to a few hundreds or even less.
For example, when you want to search for candidates from a pool of profiles stored in realtional db say Postgres, you can search by name simply using keyword, or check if the name equals to the query.
You also want to search by their profession, which you have already categorized, either by model or human labeling, this makes it a filter search.
Or if the search query requires more semantic understanding, we will leverage semantic search using embedding models.
If we want it to be more accurate, we move up to more expensive and more accurate methods such as reranking models and LLM-based retrieval methods.




Design pattern
------------------

.. figure:: /_static/retriever.png
    :align: center
    :alt: Retriever design
    :width: 620px

    How LightRAG's built-in retriever fit into the task pipeline: We focus on the high-precision retrieval method so that users can build on top of that to form the final retrieval pipeline.

LightRAG library does not prioritize the coverage of integration, the reasons are:

1. It is literally too-wide to cover them all.
2. The challenges with RAG application lies more in evaluation and optimization due to many different moving parts and many hyperparmeters, and less in implementing or integrating a 3rd party retriever.

We instead want to provide a design pattern where users can

1. Learn our existing coverage and implementation and use them out-of-the-box as building blocks.
2. Easily integrate their own retriever and make them work seamlessly with the remaining part of the LLM application so that they can evaluate and optimize the whole task pipeline.
3. Easily combine different retriever methods to form a multiple-stage retrieval pipeline.

A retriever will work hand in hand with the ``database`` and the ``data model``.
We will have a local database :class:`core.db.LocalDB` and a cloud sql-based database (using ``SQLAlchemy``) that can work with any data class, and espeically with the :class:`core.types.Document` and :class:`core.types.DialogTurn`
which provides ``context`` and ``conversation_history`` and is key to the LLM application.
As for the retriever methods, we cover the most representative methods: LLMAsRetriever, Reranker, Semantic Search, BM25, database's built-in search such as the full-text search/sql based search using Postgres and the semantic search using ``PgVector``.

.. so that users can clearly and easily integrate their own retriever, either to work withh local files or to work with cloud databases with the remaining part of an LLM application.
.. Our goal is for doing so, users get to better evaluate and optimize the whole task pipeline as a whole.

.. A retriever in our library is a component that potentially retrieves relevant ``context`` and pass it to the ``prompt`` of a ``generator``.
.. If your data is big, we assume it is users' responsibility to do fuzzy and cheap filter and search that gives high recall even though low precision till to have a manageable set of candidates (fit into local memory or a latency limit) to optimize for high precision.
.. To optimize recall, often BM25, TF-IDF, and semantic search using embedding models are used. And lastly, reranking models are used for the final precision optimization.
.. As the layer close to deliver the final user experience, we try to provide a great design pattern so that:




.. A retriever will work hand in hand with a ``database``: the retriever will be responsible for building and querying the index and work with a database, either local or cloud to save and load index.


.. A retriever will retrieve the `ids` of the ``top_k`` most relevant documents given a query. The user can then use these `ids` to retrieve the actual documents from the database.
.. The most effective approch would be ``LLMasRetriever``, ``Reranker``, ``Embedding`` + ``BM25``.

RetrieverOutput
^^^^^^^^^^^^^^^^^^^^^^^^
Retriever can require different types of input, but they all have the same output format so that we can easily switch between different retrievers in your task pipeline.

Thus our ``RetrieverOutput`` is defined as:

.. code-block:: python

    class RetrieverOutput(DataClass):
    __doc__ = r"""Save the output of a single query in retrievers."""

    doc_indices: List[int] = field(metadata={"desc": "List of document indices"})
    doc_scores: Optional[List[float]] = field(
        default=None, metadata={"desc": "List of document scores"}
    )
    query: Optional[str] = field(
        default=None, metadata={"desc": "The query used to retrieve the documents"}
    )
    documents: Optional[List[Document]] = field(
        default=None, metadata={"desc": "List of retrieved documents"}
    )


    RetrieverOutputType = List[RetrieverOutput]  # so to support multiple queries at once

You can find the types in :doc:`core.types` module.

We support both single query or a list of queries to be used as input in the retriever. The list of queries can be helpful in cases like:

(1) batch-processing, especially for semantic search where multiple queries can be represented as numpy array and be computed all at once with faster speed than doing one by one.
(2) for cases like `query expansion` where to increase the recall, users often generate multiple queries from the original query.



Retriever Base Class
^^^^^^^^^^^^^^^^^^^^^^^^

Our base class :class:`core.retriever.Retriever` highlights two parts for implementing a retriever:

1. On the algorithm side, it need to prepare the input documents and to retrieve the documents given a query.
2. On the data storage side, if the preparation stage requires heavy computation for building intermedia ``index``, it needs to communicate clearly (1) the ``index`` to users so that users can save them with their choosing storage approach,
and (2) allow users to load the index back.
so that users can easily integrate their own retriever or to customize existing ones.

**Build and Query Index -- The Algorithm**

For some retrievers, they need ``index``, which is intermediate data that is used to assist the retrieval.
.. they will compute/manage ``index`` and handles the  ``query`` applied on the index to get the relevant documents.
Index is data-structure specific to either retrieval method that is used to compute a relevancy score in the case of embeddings for semantic search and Term-Frequency-Inverse Document Frequency (TF-IDF) for BM25, and for rerankers it is just the query and the candidates files themselves and the model.
For a local retriever, it will need to (1) computes the index itself given candidates documents, persist them for later usage (2) load index from local or cloud storage (3) query the index to get the relevant documents.

The base class will have the following methods to do so:

.. code:: python

    def build_index_from_documents(
        self,
        documents: Sequence[RetrieverDocumentType],
        **kwargs,
    ):
        r"""Built index from the `text` field of each document in the list of documents.
        input_field_map_func: a function that maps the document to the input field to be used for indexing
        You can use _get_inputs to get a standard format fits for this retriever or you can write your own
        """
        raise NotImplementedError(
            f"build_index_from_documents and input_field_map_func is not implemented"
        )

    def retrieve(
        self,
        input: RetrieverInputType,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> RetrieverOutputType:
        raise NotImplementedError(f"retrieve is not implemented")

**Load and Save Index - The Data Storage**

For retriever method that has intermedia index other than the source of database data, the storage varies from retriever to retriever.
For example, ``BM25Retriever`` has the following attributes to form its index:

.. code:: python
    self.index_keys = ["nd", "t2d", "idf", "doc_len", "avgdl", "corpus_size"]


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

LLMAsRetriever
^^^^^^^^^^^^^^^^^^^^^^^^

LocalReranker
^^^^^^^^^^^^^^^^^^^^^^^^

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
   5. FAISS: https://github.com/facebookresearch/faiss
   6. Lost-in-the-middle: https://arxiv.org/abs/2104.08663 [Find the right reference]
   7. RAG: https://arxiv.org/abs/2104.08663 [Find the first paper on RAG]


.. admonition:: API References
   :class: highlight
   -
