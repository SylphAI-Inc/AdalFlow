Data
====================

    The purpose of this note is to provide an overview on data, data modeling, and data storage in LLM applications along with how LightRAG works with data.
We will conver:

* Data models on how to represent important data.
* Local database providing in-memory data management and storage.
* Examples on working with Cloud database using `fsspec` and `SQLAlchemy`.

.. note ::
    This note focus more on dealing with data needed to perform the LLM task.
    Datasets, including the input and ground truth output loading and dataset will be covered in the optimizing section.

So far, we have seen how our core components like ``Generator``, ``Embedder``, and ``Retriever`` work without any data cache/database and enforced data format to read data from and to write data to.
However, in real-world LLM applications, we can not avoid to deal with data storage:

1. Our documents to retrieve context from can be large and be stored in a file system or in a database in forms of tables or graphs.
2. We often need to pre-process a large amount of data (like text splitting and embedding and idf in BM25) in a datapipline into a cloud database.
3. We need to write records, logs to files or databases for monitoring and debugging, such as the failed llm calls.
4. When it comes to applications where states matter, like games and chatbots, we need to store the states and conversational history.


.. figure:: /_static/database.png
    :align: center
    :alt: Data model and database
    :width: 620px

    Data model and database



Data Models
--------------------

Besides of having a unified `GeneratorOutput`, `EmbedderOutput`, and `RetrieverOutput` data format,
we provide mainly :class:`core.types.Document` and :class:`core.types.DialogTurn` to help with text document and conversational histor processing and data storage.

We will explain the design of Document and DialogTurn. In this note, we will continue use these simple documents we used in the previous notes:

.. code-block:: python

    org_documents =[
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

Document
~~~~~~~~~~~~~~~
The :class:`core.types.Document` is used as Document data structure and to assist text processing in LLM applications.

1. A general document/text container with fields ``text``, ``meta_data``, and ``id``.
2. Assist text splitting with fields ``parent_doc_id`` and ``order``.
3. Assist embedding with fields ``vector``.
4. Assist using it as a prompt for LLM with fields ``estimated_num_tokens``.

This is why data processing components like ``TextSplitter`` and ``ToEmbeddings``  requires ``Document`` as input of each data item.

**Create a Document**

.. code-block:: python

    from lightrag.core.types import Document

    documents  = [Document(text=doc['content'], meta_data={'title': doc['title']}) for doc in org_documents]
    print(documents)

The printout will be:

.. code-block::

    [Document(id=73c12be3-7844-435b-8678-2e8e63041698, text='Renewable energy technologies not only help in reducing greenhouse gas emissions but also contribute...', meta_data={'title': 'The Impact of Renewable Energy on the Economy'}, vector=[], parent_doc_id=None, order=None, score=None), Document(id=7a17ed45-569a-4206-9670-5316efd58d58, text='Solar panels convert sunlight into electricity by allowing photons, or light particles, to knock ele...', meta_data={'title': 'Understanding Solar Panels'}, vector=[], parent_doc_id=None, order=None, score=None), Document(id=bcbc6ff9-518a-405a-8b0d-840021aa1953, text='While solar energy offers substantial environmental benefits, such as reducing carbon footprints and...', meta_data={'title': 'Pros and Cons of Solar Energy'}, vector=[], parent_doc_id=None, order=None, score=None), Document(id=ec910402-f98f-4077-a958-7335e34ee0c6, text='Renewable energy sources like wind, solar, and hydro power play a crucial role in combating climate ...', meta_data={'title': 'Renewable Energy and Its Effects'}, vector=[], parent_doc_id=None, order=None, score=None)]


DialogTurn
~~~~~~~~~~~~~~~~~~
The :class:`core.types.DialogTurn` is only used as a data structure to a user-assistant conversation turn in LLM applications.
**If we need to apply a text processing pipeline to a conversational history, we will convert it to ``Document`` first.**

.. note ::
    For both ``Document`` and ``DialogTurn``, we have an equivalent class in :doc:`database.sqlalchemy.model` to handle the persitence of data in a SQL database.

Data Pipeline
--------------------
Let's see how to can write a data pipeline that can process any form of text data by using intermediate data model-``Document``.
We will use ``ord_documents`` and a list of ``DialogTurn`` as examples. As our data pipelines are designed to work with ``Document`` structure,
we simplify just need to add a mapping function to convert the original data to ``Document``.

.. code-block:: python

    # mapping function for org_documents
    def map_to_document(doc: Dict) -> Document:
        return Document(text=doc['content'], meta_data={'title': doc['title']})

    # mapping function for dialog_turns
    def map_dialogturn_to_document(turn: DialogTurn) -> Document:
        # it can be important to keep the original data's id
        return Document(id=turn.id, text=turn.user_query + ' ' + turn.assistant_response)

**Text Splitting**

**Embedding**




Local database
--------------------
In-memory management and storage of ``Document`` and ``DialogTurn`` objects are provided by :class:`core.database.DocumentDatabase` and :class:`core.database.DialogDatabase` respectively.

**Data Loading and CRUD Operations**

**Data Processing/Transformation Pipeline(such as TextSplitter and Embedder)**

**Save/Persistence**

**Use With Retriever**

**Use With Generator**


Cloud database
--------------------

Suggestion on File reading and writing
------------------------------------------
We dont provide integration on using ``fsspec``, but here we can give you some suggestions on how to use it.



Graph database
--------------------


.. admonition:: API References
   :class: highlight

   - :class:`core.types.Document`
   - :class:`core.types.DialogTurn`
   - :class:`core.db.LocalDB`
   - :class:`core.text_splitter.DocumentSplitter`
   - :class:`core.data_components.ToEmbeddings`
