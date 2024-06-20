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


There are numerous search techniques long existing before the vector/semantic search and reranking models, such as keyword search, fuzzy search, proximity search, phrase search, boolean search, facet search, full-text search,
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

**Retrieval in Production**

In real production, retrieval is often of multiple-stages, from the cheapest to the most expensive and most accurate, from millions of candidates to a few hundreds or even less.
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

    LightRAG retriever covers high-precision retriever methods and make them work locally and in-memory, this will help researchers and developers build and test.
    We also showcase how it is like to work with cloud database for **large-scale data** along with its built-in search& filter methods.


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
As for the retriever methods, we cover the most representative methods: LLMAsRetriever, Reranker(Cross-encoder), Semantic Search(Bi-encoder), BM25, database's built-in search such as the full-text search/sql based search using Postgres and the semantic search using ``PgVector``.

.. so that users can clearly and easily integrate their own retriever, either to work withh local files or to work with cloud databases with the remaining part of an LLM application.
.. Our goal is for doing so, users get to better evaluate and optimize the whole task pipeline as a whole.

.. A retriever in our library is a component that potentially retrieves relevant ``context`` and pass it to the ``prompt`` of a ``generator``.
.. If your data is big, we assume it is users' responsibility to do fuzzy and cheap filter and search that gives high recall even though low precision till to have a manageable set of candidates (fit into local memory or a latency limit) to optimize for high precision.
.. To optimize recall, often BM25, TF-IDF, and semantic search using embedding models are used. And lastly, reranking models are used for the final precision optimization.
.. As the layer close to deliver the final user experience, we try to provide a great design pattern so that:




.. A retriever will work hand in hand with a ``database``: the retriever will be responsible for building and querying the index and work with a database, either local or cloud to save and load index.


.. A retriever will retrieve the `ids` of the ``top_k`` most relevant documents given a query. The user can then use these `ids` to retrieve the actual documents from the database.
.. The most effective approch would be ``LLMasRetriever``, ``Reranker``, ``Embedding`` + ``BM25``.

Retriever Data Types
^^^^^^^^^^^^^^^^^^^^^^^^
In most cases, the query is string. But there are cases we might need both text and images as a query, such as "find me a cloth that looks like this".
We defined the query type as:

.. code-block:: python

    RetrieverQueryType = TypeVar("RetrieverQueryType", contravariant=True)
    RetrieverStrQueryType = str
    RetrieverQueriesType = Union[RetrieverQueryType, Sequence[RetrieverQueryType]]
    RetrieverStrQueriesType = Union[str, Sequence[RetrieverStrQueryType]]

As we see, our retriever should be able to handle both single query and multiple queries at once.

The documents are a sequence of document of any type that will be later specified by the subclass:

.. code-block:: python

    RetrieverDocumentType = TypeVar("RetrieverDocumentType", contravariant=True) # a single document
    RetrieverDocumentsType = Sequence[RetrieverDocumentType] # The final documents types retriever can use


We further define  the same output format so that we can easily switch between different retrievers in our task pipeline.
Here is our output format:


.. code-block:: python

    class RetrieverOutput(DataClass):
        __doc__ = r"""Save the output of a single query in retrievers.

        It is up to the subclass of Retriever to specify the type of query and document.
        """

        doc_indices: List[int] = field(metadata={"desc": "List of document indices"})
        doc_scores: Optional[List[float]] = field(
            default=None, metadata={"desc": "List of document scores"}
        )
        query: Optional[RetrieverQueryType] = field(
            default=None, metadata={"desc": "The query used to retrieve the documents"}
        )
        documents: Optional[List[RetrieverDocumentType]] = field(
            default=None, metadata={"desc": "List of retrieved documents"}
        )


    RetrieverOutputType = List[RetrieverOutput]  # so to support multiple queries at once

You can find the types in :ref:`core.types<core-types>`. The list of queries and `RetrieverOutput` can be helpful for:

(1) Batch-processing: especially for semantic search where multiple queries can be represented as numpy array and be computed all at once with faster speed than doing one by one.
(2) For `query expansion` where to increase the recall, users often generate multiple queries from the original query.


.. code-block:: python

    class Retriever(Component, Generic[RetrieverDocumentType, RetrieverQueryType]):

        ...

        def call(
            self,
            input: RetrieverQueriesType,
            top_k: Optional[int] = None,
            **kwargs,
        ) -> RetrieverOutputType:
            raise NotImplementedError(f"retrieve is not implemented")

        async def acall(
            self,
            input: RetrieverQueriesType,
            top_k: Optional[int] = None,
            **kwargs,
        ) -> RetrieverOutputType:
            raise NotImplementedError(f"Async retrieve is not implemented")


**Document and TextSplitter**

If your documents(text format) are too large and it is a common practise to first use ``TextSplitter`` to split them into smaller chunks.
Please refer to :doc:`text_splitter` and our provided notebook on how to use it.



Retriever Base Class
^^^^^^^^^^^^^^^^^^^^^^^^

Functionally, the base retriever :class:`core.retriever.Retriever` defines another required method ``build_index_from_documents`` where the subclass will prepare the retriever for the actual retrieval calls.
Optionally, the subclass can implement ``save_to_file`` and ``load_from_file`` to save and load the retriever to/from disk.
As the retriever is a subclass of component, you already inherited powerful serialization and deserialization methods such as ``to_dict``, ``from_dict``, and ``from_config`` to help
with the saving and loading process. As for helper attributes, we have ``indexed`` and ``index_keys`` to differentiate if the retriever is ready for retrieval and the attributes that are key to restore the functionality/states of the retriever.
It is up the subclass to decide how to decide the storage of the index, it can be in-memory, local disk, or cloud storage, or save as json or pickle file or even a db table.
As an example, :class:`components.retriever.bm25_retriever.InMemoryBM25Retriever` has the following key attributes to index.

.. code:: python

    self.index_keys = ["nd", "t2d", "idf","doc_len","avgdl","total_documents","top_k","k1","b","epsilon","indexed"]


Retriever in Action
--------------------
All of our retrievers are  subclassed from the base retriever, and they are located in the ``components.retriever`` module.
You can skim through their implementations here: :ref:`retriever<components-retriever>`.
Currently only :class:`components.retriever.faiss_retriever.InMemoryBM25Retriever` needs to have its own ``save_to_file`` and ``load_from_file`` to avoid recomputation again.
The ``FAISSRetriever`` will work with a database instead to store the embeddings and it alleviates the need for the retriever to deal with states saving.

In this note, we will use the following documents and queries for demonstration:

.. code-block:: python

    query_1 = "What are the benefits of renewable energy?" # gt is [0, 3]
    query_2 = "How do solar panels impact the environment?" # gt is [1, 2]

    documents =[
        {
            "title": "The Impact of Renewable Energy on the Economy",
            "content": "Renewable energy technologies not only help in reducing greenhouse gas emissions but also contribute significantly to the economy by creating jobs in the manufacturing and installation sectors. The growth in renewable energy usage boosts local economies through increased investment in technology and infrastructure."
        },
        {
            "title": "Understanding Solar Panels",
            "content": "Solar panels convert sunlight into electricity by allowing photons, or light particles, to knock electrons free from atoms, generating a flow of electricity. Solar panels are a type of renewable energy technology that has been found to have a significant positive effect on the environment by reducing the reliance on fossil fuels."
        },
        {
            "title": "Pros and Cons of Solar Energy",
            "content": "While solar energy offers substantial environmental benefits, such as reducing carbon footprints and pollution, it also has downsides. The production of solar panels can lead to hazardous waste, and large solar farms require significant land, which can disrupt local ecosystems."
        },
        {
            "title":  "Renewable Energy and Its Effects",
            "content": "Renewable energy sources like wind, solar, and hydro power play a crucial role in combating climate change. They do not produce greenhouse gases during operation, making them essential for sustainable development. However, the initial setup and material sourcing for these technologies can still have environmental impacts."
        }
    ]

The first query should retrieve the first and the last document, and the second query should retrieve the second and the third document.

In-memory FAISSRetriever
^^^^^^^^^^^^^^^^^^^^^^^^
First, let's do semantic search, here we will use in-memory :class:`components.retriever.faiss_retriever.InMemoryFAISSRetriever`.
FAISS retriever takes embeddings which can be ``List[float]`` or ``np.ndarray`` and build an index using FAISS library.
The query can take both embeddings and str formats.

.. note ::
    ``faiss`` package is optional in our library. When you want to use it, ensure you have it installed in your env.

We will quickly prepare the embeddings of the above documents using `content` field.

.. code-block:: python

    from lightrag.core.embedder import Embedder
    from lightrag.core.types import ModelClientType


    model_kwargs = {
        "model": "text-embedding-3-small",
        "dimensions": 256,
        "encoding_format": "float",
    }

    embedder = Embedder(model_client =ModelClientType.OPENAI(), model_kwargs=model_kwargs)
    output = embedder(input=[doc["content"] for doc in documents])
    documents_embeddings = [x.embedding for x in output.data]


For the initialization, a retriever can take both its required documents along with hyperparmeters including ``top_k``.
The ``documents`` field is optional. Let's pass it all from ``__init__`` first:

.. code-block:: python

    from lightrag.components.retriever import FAISSRetriever
    retriever = FAISSRetriever(top_k=2, embedder=embedder, documents=documents_embeddings)

    print(retriever)

The printout:

.. code-block::

    FAISSRetriever(
     top_k=2, metric=prob, dimensions=256, total_documents=4
     (embedder): Embedder(
        model_kwargs={'model': 'text-embedding-3-small', 'dimensions': 256, 'encoding_format': 'float'},
        (model_client): OpenAIClient()
     )
    )

We can also pass the documents using :meth:`components.retriever.faiss_retriever.InMemoryFAISSRetriever.build_index_from_documents` method after the initialization.
This is helpful when your retriever would need to work with different pool of documents each time.

.. code-block:: python

    retriever_1 = FAISSRetriever(top_k=2, embedder=embedder)
    retriever_1.build_index_from_documents(documents=documents_embeddings)

Now, we will do the retriever, the input can either be a single query or a list of queries:

.. code-block:: python

    output_1 = retriever(input=query_1)
    output_2 = retriever(input=query_2)
    output_3 = retriever(input = [query_1, query_2])
    print(output_1)
    print(output_2)
    print(output_3)

The printout is:

.. code-block::

    [RetrieverOutput(doc_indices=[0, 3], doc_scores=[0.8119999766349792, 0.7749999761581421], query='What are the benefits of renewable energy?', documents=None)]
    [RetrieverOutput(doc_indices=[2, 1], doc_scores=[0.8169999718666077, 0.8109999895095825], query='How do solar panels impact the environment?', documents=None)]
    [RetrieverOutput(doc_indices=[0, 3], doc_scores=[0.8119999766349792, 0.7749999761581421], query='What are the benefits of renewable energy?', documents=None), RetrieverOutput(doc_indices=[2, 1], doc_scores=[0.8169999718666077, 0.8109999895095825], query='How do solar panels impact the environment?', documents=None)]

In default, the score is a simulated probabity in range ``[0, 1]`` using consine similarity. The higher the score, the more relevant the document is to the query.
You can check the retriever for more type of scores.

BM25Retriever
^^^^^^^^^^^^^^^^^^^^^^^^
So the semantic search works pretty well. We will see how :class:`components.retriever.bm25_retriever.InMemoryBM25Retriever` works in comparison.
We reimplemented the code in [9]_ with one improvement: instead of using ``text.split(" ")``, we use tokenizer to split the text. Here is a comparison of how they different:

.. code-block:: python

    from lightrag.components.retriever.bm25_retriever import split_text_by_word_fn_then_lower_tokenized, split_text_by_word_fn

    query_1_words = split_text_by_word_fn(query_1)
    query_1_tokens = split_text_by_word_fn_then_lower_tokenized(query_1)

Output:

.. code-block::

    ['what', 'are', 'the', 'benefits', 'of', 'renewable', 'energy?']
    ['what', 'are', 'the', 'benef', 'its', 'of', 're', 'new', 'able', 'energy', '?']

We prepare the retriever:

.. code-block:: python

    from lightrag.components.retriever import InMemoryBM25Retriever

    document_map_func = lambda x: x["content"]

    bm25_retriever = InMemoryBM25Retriever(top_k=2, documents=documents, document_map_func=document_map_func)
    print(bm25_retriever)

It takes ``document_map_func`` to map the documents to the text format the retriever can work with.
The output is:

.. code-block::

    InMemoryBM25Retriever(top_k=2, k1=1.5, b=0.75, epsilon=0.25, use_tokenizer=True, total_documents=4)

Now we call the retriever exactly the same way as we did with the FAISS retriever:

.. code-block:: python

    output_1 = bm25_retriever(input=query_1)
    output_2 = bm25_retriever(input=query_2)
    output_3 = bm25_retriever(input = [query_1, query_2])
    print(output_1)
    print(output_2)
    print(output_3)

The printout is:

.. code-block::

    [RetrieverOutput(doc_indices=[2, 1], doc_scores=[2.151683837681807, 1.6294762236217233], query='What are the benefits of renewable energy?', documents=None)]
    [RetrieverOutput(doc_indices=[3, 2], doc_scores=[1.5166601493236314, 0.7790170272403408], query='How do solar panels impact the environment?', documents=None)]
    [RetrieverOutput(doc_indices=[2, 1], doc_scores=[2.151683837681807, 1.6294762236217233], query='What are the benefits of renewable energy?', documents=None), RetrieverOutput(doc_indices=[3, 2], doc_scores=[1.5166601493236314, 0.7790170272403408], query='How do solar panels impact the environment?', documents=None)]

Here we see the first query returns ``[2, 1]`` while the ground truth is ``[0, 3]``. The second query returns ``[3, 2]`` while the ground truth is ``[1, 2]``.
The performance is quite disappointing. BM25 is known for lack of semantic understanding and does not consider context.
We tested on the shorter and almost key-word like version of our queries and use both the `title` and `content`, and it gives the right response using the tokenized split.

.. code-block:: python

    query_1_short = "renewable energy?"  # gt is [0, 3]
    query_2_short = "solar panels?"  # gt is [1, 2]
    document_map_func = lambda x: x["title"] + " " + x["content"]
    bm25_retriever.build_index_from_documents(documents=documents, document_map_func=document_map_func)

This time the retrieval gives us the right answer.

.. code-block::

    [RetrieverOutput(doc_indices=[0, 3], doc_scores=[0.9498793313012154, 0.8031794089550072], query='renewable energy?', documents=None)]
    [RetrieverOutput(doc_indices=[2, 1], doc_scores=[0.5343238380789569, 0.4568096570283078], query='solar panels?', documents=None)]

Reranker as Retriever
^^^^^^^^^^^^^^^^^^^^^^^^
Semantic search works well, and reranker basd on mostly `cross-encoder` model is supposed to work even better.
We have integrated two rerankers: ``BAAI/bge-reranker-base`` [10]_ hosted on ``transformers`` and rerankers provided by ``Cohere`` [11]_.
These models follow the ``ModelClient`` protocol and are directly accessible as retriever from :class:`components.retriever.reranker_retriever.RerankerRetriever`.




**Reranker ModelClient Integration**

A reranker will take ``ModelType.RERANKER`` and the standard LightRAG library requires it to have four arguments in the ``model_kwargs``:
``['model', 'top_k', 'documents', 'query']``. It is in the ModelClient which converts LightRAG's standard arguments to the model's specific arguments.
If you want to intergrate your reranker, either locally or using APIs, check out :class:`components.model_client.transformers_client.TransformersClient` and
:class:`components.model_client.cohere_client.CohereAPIClient` for how to do it.


To use it from the ``RerankerRetriever``, we only need to pass the ``model`` along with other arguments who does not
require conversion in the ``model_kwargs``. Here is how we use model  `rerank-english-v3.0` from Cohere(Make sure you have the cohere sdk installed and prepared your api key):

.. code-block:: python

    from lightrag.components.retriever import RerankerRetriever

    model_client = ModelClientType.COHERE()
    model_kwargs = {"model": "rerank-english-v3.0"}


    reranker = RerankerRetriever(
        top_k=2, model_client=model_client, model_kwargs=model_kwargs
    )
    print(reranker)

The printout:

.. code-block::

    RerankerRetriever(
        top_k=2, model_kwargs={'model': 'rerank-english-v3.0'}, model_client=CohereAPIClient(), total_documents=0
        (model_client): CohereAPIClient()
    )

Now we build the index and do the retrieval:


.. code-block:: python

    document_map_func = lambda x: x["content"]
    reranker.build_index_from_documents(documents=documents, document_map_func=document_map_func)

    output_1 = reranker(input=query_1)
    output_2 = reranker(input=query_2)
    output_3 = reranker(input = [query_1, query_2])

From the structure after adding documents we see the reranker has passed the documents to the ``model_kwargs`` so that it can send it all to the ``ModelClient``.

.. code-block::

    RerankerRetriever(
        top_k=2, model_kwargs={'model': 'rerank-english-v3.0', 'documents': ['Renewable energy technologies not only help in reducing greenhouse gas emissions but also contribute significantly to the economy by creating jobs in the manufacturing and installation sectors. The growth in renewable energy usage boosts local economies through increased investment in technology and infrastructure.', 'Solar panels convert sunlight into electricity by allowing photons, or light particles, to knock electrons free from atoms, generating a flow of electricity. Solar panels are a type of renewable energy technology that has been found to have a significant positive effect on the environment by reducing the reliance on fossil fuels.', 'While solar energy offers substantial environmental benefits, such as reducing carbon footprints and pollution, it also has downsides. The production of solar panels can lead to hazardous waste, and large solar farms require significant land, which can disrupt local ecosystems.', 'Renewable energy sources like wind, solar, and hydro power play a crucial role in combating climate change. They do not produce greenhouse gases during operation, making them essential for sustainable development. However, the initial setup and material sourcing for these technologies can still have environmental impacts.']}, model_client=CohereAPIClient(), total_documents=4
        (model_client): CohereAPIClient()
    )

From the results we see it gets the right answer and has a close to 1 score.

.. code-block::

    [RetrieverOutput(doc_indices=[0, 3], doc_scores=[0.99520767, 0.9696708], query='What are the benefits of renewable energy?', documents=None)]
    [RetrieverOutput(doc_indices=[1, 2], doc_scores=[0.98742366, 0.9701269], query='How do solar panels impact the environment?', documents=None)]
    [RetrieverOutput(doc_indices=[0, 3], doc_scores=[0.99520767, 0.9696708], query='What are the benefits of renewable energy?', documents=None), RetrieverOutput(doc_indices=[1, 2], doc_scores=[0.98742366, 0.9701269], query='How do solar panels impact the environment?', documents=None)]

Now let us see how the ``BAAI/bge-reranker-base` from local transformers model works:

.. code-block:: python

    model_client = ModelClientType.TRANSFORMERS()
    model_kwargs = {"model": "BAAI/bge-reranker-base"}

    reranker = RerankerRetriever(
        top_k=2,
        model_client=model_client,
        model_kwargs=model_kwargs,
        documents=documents,
        document_map_func=document_map_func,
    )
    print(reranker)

The printout:

.. code-block::

    RerankerRetriever(
        top_k=2, model_kwargs={'model': 'BAAI/bge-reranker-base', 'documents': ['Renewable energy technologies not only help in reducing greenhouse gas emissions but also contribute significantly to the economy by creating jobs in the manufacturing and installation sectors. The growth in renewable energy usage boosts local economies through increased investment in technology and infrastructure.', 'Solar panels convert sunlight into electricity by allowing photons, or light particles, to knock electrons free from atoms, generating a flow of electricity. Solar panels are a type of renewable energy technology that has been found to have a significant positive effect on the environment by reducing the reliance on fossil fuels.', 'While solar energy offers substantial environmental benefits, such as reducing carbon footprints and pollution, it also has downsides. The production of solar panels can lead to hazardous waste, and large solar farms require significant land, which can disrupt local ecosystems.', 'Renewable energy sources like wind, solar, and hydro power play a crucial role in combating climate change. They do not produce greenhouse gases during operation, making them essential for sustainable development. However, the initial setup and material sourcing for these technologies can still have environmental impacts.']}, model_client=TransformersClient(), total_documents=4
        (model_client): TransformersClient()
    )

Here is the retrieval result:

.. code-block::

    [RetrieverOutput(doc_indices=[0, 3], doc_scores=[0.9996004700660706, 0.9950029253959656], query='What are the benefits of renewable energy?', documents=None)]
    [RetrieverOutput(doc_indices=[2, 0], doc_scores=[0.9994490742683411, 0.9994476437568665], query='How do solar panels impact the environment?', documents=None)]

It missed one at the second query, but it is at the top 3.
Semantically,  these documents might be close.
If we use top_k = 3, the genearator might be able to filter out the irrelevant one and eventually give out the right final response.
Also, if we use both the `title` and `content`, it will also got the right response.



LLM as Retriever
^^^^^^^^^^^^^^^^^^^^^^^^

There are differen ways to use LLM as a retriever:

1. Directly show it of all documents and query and ask it to return the indices of the top_k as a list.
2. Put the query and document a pair and ask it to do a `yes` and `no`. Additionally, we can use its `logprobs` of the `yes` token to get a probability-like score. We will implement this in the near future, for now, you can refer [8]_ to implement it yourself.

For the first case, with out prompt and zero-shot, `gpt-3.5-turbo` is not working as well as `gpt-4o` which got both answers right.
Here is our code:

.. code-block:: python

    from lightrag.components.retriever import LLMRetriever

    model_client = ModelClientType.OPENAI()
    model_kwargs = {
        "model": "gpt-4o",
    }
    document_map_func = lambda x: x["content"]
    llm_retriever = LLMRetriever(
            top_k=2,
            model_client=model_client,
            model_kwargs=model_kwargs,
            documents=documents,
            document_map_func=document_map_func
        )
    print(llm_retriever)

The printout:

.. code-block::

    LLMRetriever(
        top_k=2, total_documents=4,
        (generator): Generator(
            model_kwargs={'model': 'gpt-4o'},
            (prompt): Prompt(
            template: <SYS>
            You are a retriever. Given a list of documents, you will retrieve the top_k {{top_k}} most relevant documents and output the indices (int) as a list:
            [<index of the most relevant with top_k options>]
            <Documents>
            {% for doc in documents %}
            ```Index {{ loop.index - 1 }}. {{ doc }}```
            {% endfor %}
            </Documents>
            </SYS>
            Query: {{ input_str }}
            You:
            , preset_prompt_kwargs: {'top_k': 2, 'documents': ['Renewable energy technologies not only help in reducing greenhouse gas emissions but also contribute significantly to the economy by creating jobs in the manufacturing and installation sectors. The growth in renewable energy usage boosts local economies through increased investment in technology and infrastructure.', 'Solar panels convert sunlight into electricity by allowing photons, or light particles, to knock electrons free from atoms, generating a flow of electricity. Solar panels are a type of renewable energy technology that has been found to have a significant positive effect on the environment by reducing the reliance on fossil fuels.', 'While solar energy offers substantial environmental benefits, such as reducing carbon footprints and pollution, it also has downsides. The production of solar panels can lead to hazardous waste, and large solar farms require significant land, which can disrupt local ecosystems.', 'Renewable energy sources like wind, solar, and hydro power play a crucial role in combating climate change. They do not produce greenhouse gases during operation, making them essential for sustainable development. However, the initial setup and material sourcing for these technologies can still have environmental impacts.']}, prompt_variables: ['documents', 'top_k', 'input_str']
            )
            (model_client): OpenAIClient()
            (output_processors): ListParser()
        )
    )

Here is the response:

.. code-block::

    [RetrieverOutput(doc_indices=[0, 3], doc_scores=None, query='What are the benefits of renewable energy?', documents=None)]
    [RetrieverOutput(doc_indices=[1, 2], doc_scores=None, query='How do solar panels impact the environment?', documents=None)]

We can call the retriever with different model without reinitializing the retriever. Here is how we do it with `gpt-3.5-turbo`:

.. code-block:: python

    model_kwargs = {
        "model": "gpt-3.5-turbo",
    }
    output_1 = llm_retriever(model_kwargs=model_kwargs, input=query_1)
    output_2 = llm_retriever(model_kwargs=model_kwargs, input=query_2)

The response is:

.. code-block::

    [RetrieverOutput(doc_indices=[0, 1], doc_scores=None, query='What are the benefits of renewable energy?', documents=None)]
    [RetrieverOutput(doc_indices=[1, 2], doc_scores=None, query='How do solar panels impact the environment?', documents=None)]


PostgresRetriever
^^^^^^^^^^^^^^^^^^^^^^^^


Use together with Database
-----------------------------
When the scale of data is large, we will use a database to store the computed embeddings and indexes from the documents.

With LocalDB
^^^^^^^^^^^^^^^^^^^^^^^^
We have previously computed embeddings, now let us :class:`core.db.LocalDB` to help with the persistence.
(Although you can totally persist them yourself such as using pickle).
Additionally, ``LocalDB`` help us keep track of our initial documents and its transformed documents.


.. admonition:: References
   :class: highlight

   1. Full-text search on PostgreSQL: https://www.postgresql.org/docs/current/textsearch.html
   2. BM25: https://en.wikipedia.org/wiki/Okapi_BM25
   3. Representative learning models: https://arxiv.org/abs/2104.08663 [Find the right reference]
   4. Reranking models: https://arxiv.org/abs/2104.08663 [Find the right reference]
   5. FAISS: https://github.com/facebookresearch/faiss
   6. Lost-in-the-middle: https://arxiv.org/abs/2104.08663 [Find the right reference]
   7. RAG: https://arxiv.org/abs/2104.08663 [Find the first paper on RAG]
   8. Use LLM as Reranker along with logprobs: https://cookbook.openai.com/examples/search_reranking_with_cross-encoders/
   9. Rank_bm25: https://github.com/dorianbrown/rank_bm25
   10. https://huggingface.co/BAAI/bge-reranker-base
   11. Cohere reranker: https://docs.cohere.com/reference/rerank


.. admonition:: API References
   :class: highlight

   - :class:`core.retriever.Retriever`
   - :ref:`core.types<core-types>`
   - :class:`components.retriever.faiss_retriever.FAISSRetriever`
   - :class:`components.retriever.bm25_retriever.InMemoryBM25Retriever`
   - :class:`components.retriever.reranker_retriever.RerankerRetriever`
   - :class:`components.retriever.llm_retriever.LLMRetriever`
