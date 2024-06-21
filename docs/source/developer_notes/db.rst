Data
====================

    The purpose of this note is to provide an overview on data, data modeling, and data storage in LLM applications along with how LightRAG works with data.
We will conver:

* Data models on how to represent important data.
* Data pipeline to process data in LLM applications.
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

    turns = [
        {
            "user": "What are the benefits of renewable energy?",
            "system": "I can see you are interested in renewable energy. Renewable energy technologies not only help in reducing greenhouse gas emissions but also contribute significantly to the economy by creating jobs in the manufacturing and installation sectors. The growth in renewable energy usage boosts local economies through increased investment in technology and infrastructure.",
            "user_time": "2021-09-01T12:00:00Z",
            "system_time": "2021-09-01T12:00:01Z"
        },
        {
            "user": "How do solar panels impact the environment?",
            "system": "Solar panels convert sunlight into electricity by allowing photons, or light particles, to knock electrons free from atoms, generating a flow of electricity. Solar panels are a type of renewable energy technology that has been found to have a significant positive effect on the environment by reducing the reliance on fossil fuels.",
            "user_time": "2021-09-01T12:00:02Z",
            "system_time": "2021-09-01T12:00:03Z"
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
**If we need to apply a text processing pipeline to a conversational history, we will use our text container``Document`` to store the text we need to use.**

.. note ::
    For both ``Document`` and ``DialogTurn``, we have an equivalent class in :doc:`database.sqlalchemy.model`(:class:`database.sqlalchemy.modoel.Document`) to handle the persitence of data in a SQL database.

Here is how to get a list of ``DialogTurn`` from the ``turns``:

.. code-block:: python

    from lightrag.core.types import DialogTurn

    dialog_turns = [DialogTurn(user_query = turn['user'], assistant_response = turn['system'], user_query_timestamp = turn['user_time'], assistant_response_timestamp = turn['system_time']) for turn in turns]
    print(dialog_turns)

The printout will be:

.. code-block::

    [DialogTurn(id='e3b48bcc-df68-43a4-aa81-93922b619293', user_id=None, session_id=None, order=None, user_query='What are the benefits of renewable energy?', assistant_response='I can see you are interested in renewable energy. Renewable energy technologies not only help in reducing greenhouse gas emissions but also contribute significantly to the economy by creating jobs in the manufacturing and installation sectors. The growth in renewable energy usage boosts local economies through increased investment in technology and infrastructure.', user_query_timestamp='2021-09-01T12:00:00Z', assistant_response_timestamp='2021-09-01T12:00:01Z', metadata=None, vector=None), DialogTurn(id='21f0385d-d19a-442f-ae99-910e984cdb65', user_id=None, session_id=None, order=None, user_query='How do solar panels impact the environment?', assistant_response='Solar panels convert sunlight into electricity by allowing photons, or light particles, to knock electrons free from atoms, generating a flow of electricity. Solar panels are a type of renewable energy technology that has been found to have a significant positive effect on the environment by reducing the reliance on fossil fuels.', user_query_timestamp='2021-09-01T12:00:02Z', assistant_response_timestamp='2021-09-01T12:00:03Z', metadata=None, vector=None)]


Data Pipeline
--------------------
Let's see how to can write a data pipeline that can process any form of text data by using intermediate data model-``Document``.
Currently, we have two data processing components: ``TextSplitter`` and ``ToEmbeddings`` in the ``components.data_process`` module.

We will use ``ord_documents`` and a list of ``DialogTurn`` as examples. As our data pipelines are designed to work with ``Document`` structure,
we simplify just need to add a mapping function to convert the original data to ``Document``.

.. code-block:: python

    # mapping function for org_documents
    def map_to_document(doc: Dict) -> Document:
        return Document(text=doc['content'], meta_data={'title': doc['title']})

    def map_dialogturn_to_document(turn: DialogTurn) -> Document:
        # it can be important to keep the original data's id
        return Document(id=turn.id, text=turn.user_query + ' ' + turn.assistant_response)

You can refer to :doc:`text_splitter` for more details on how to use ``TextSplitter``.
``ToEmbeddings`` is an orchestrator on ``BatchEmbedder`` and it will generate embeddings for a list of ``Document`` and store the embeddings as ``List[Float]`` in the ``vector`` field of each ``Document``.

``Sequential`` can be easily used to chain multiple data processing components together.
Here is the code to form a data pipeline:

.. code-block:: python

    from lightrag.core.embedder import Embedder
    from lightrag.core.types import ModelClientType
    from lightrag.components.data_process import DocumentSplitter, ToEmbeddings
    from lightrag.core.component import Sequential


    model_kwargs = {
        "model": "text-embedding-3-small",
        "dimensions": 256,
        "encoding_format": "float",
    }

    splitter_config = {
        "split_by": "word",
        "split_length": 50,
        "split_overlap": 10
    }

    splitter = DocumentSplitter(**splitter_config)
    embedder = Embedder(model_client =ModelClientType.OPENAI(), model_kwargs=model_kwargs)
    embedder_transformer = ToEmbeddings(embedder, batch_size=2)
    data_transformer = Sequential(splitter, embedder_transformer)
    print(data_transformer)

The printout will be:

.. code-block::

    Sequential(
    (0): DocumentSplitter(split_by=word, split_length=50, split_overlap=10)
    (1): ToEmbeddings(
        batch_size=2
        (embedder): Embedder(
        model_kwargs={'model': 'text-embedding-3-small', 'dimensions': 256, 'encoding_format': 'float'},
        (model_client): OpenAIClient()
        )
        (batch_embedder): BatchEmbedder(
        (embedder): Embedder(
            model_kwargs={'model': 'text-embedding-3-small', 'dimensions': 256, 'encoding_format': 'float'},
            (model_client): OpenAIClient()
        )
        )
        )
    )

Now, apply the data pipeline to the ``dialog_turns``:

.. code-block:: python

    dialog_turns_as_documents = [map_dialogturn_to_document(turn) for turn in dialog_turns]
    print(dialog_turns_as_documents)

    # apply data transformation to the documents
    output = data_transformer(dialog_turns_as_documents)
    print(output)

The printout will be:

.. code-block::

    [Document(id=e3b48bcc-df68-43a4-aa81-93922b619293, text='What are the benefits of renewable energy? I can see you are interested in renewable energy. Renewab...', meta_data=None, vector=[], parent_doc_id=None, order=None, score=None), Document(id=21f0385d-d19a-442f-ae99-910e984cdb65, text='How do solar panels impact the environment? Solar panels convert sunlight into electricity by allowi...', meta_data=None, vector=[], parent_doc_id=None, order=None, score=None)]
    Splitting documents: 100%|██████████| 2/2 [00:00<00:00, 609.37it/s]
    Batch embedding documents: 100%|██████████| 2/2 [00:00<00:00,  3.79it/s]
    Adding embeddings to documents from batch: 2it [00:00, 10205.12it/s]
    [Document(id=e636facc-8bc3-483b-afbd-37e1d8ff0526, text='What are the benefits of renewable energy? I can see you are interested in renewable energy. Renewab...', meta_data=None, vector='len: 256', parent_doc_id=e3b48bcc-df68-43a4-aa81-93922b619293, order=0, score=None), Document(id=06ea7cea-c4e4-4f5f-b3e9-2e6f4452827b, text='and installation sectors. The growth in renewable energy usage boosts local economies through increa...', meta_data=None, vector='len: 256', parent_doc_id=e3b48bcc-df68-43a4-aa81-93922b619293, order=1, score=None), Document(id=0018af12-c8fc-49ff-ab64-a2acf8ba4c27, text='How do solar panels impact the environment? Solar panels convert sunlight into electricity by allowi...', meta_data=None, vector='len: 256', parent_doc_id=21f0385d-d19a-442f-ae99-910e984cdb65, order=0, score=None), Document(id=c5431397-2a78-4870-abce-353b738c1b71, text='has been found to have a significant positive effect on the environment by reducing the reliance on ...', meta_data=None, vector='len: 256', parent_doc_id=21f0385d-d19a-442f-ae99-910e984cdb65, order=1, score=None)]



Local database
--------------------
:class:`core.db.LocalDB` offers in-memory CRUD data operations, data transfomation/processing pipelines and processed data
In-memory management and storage of ``Document`` and ``DialogTurn`` objects are provided by :class:`core.db.LocalDB`.

**LocalDB class**

``LocalDB`` is a container to store and manage a sequence of items of any data type.


**Data Loading and CRUD Operations**

Let's create a ``LocalDB`` to manage the ``dialog_turns``:

.. code-block:: python

    from lightrag.core.db import LocalDB

    dialog_turn_db = LocalDB('dialog_turns')
    print(dialog_turn_db)

    dialog_turn_db.load(dialog_turns)
    print(dialog_turn_db)

The printout will be:

.. code-block::

    LocalDB(name='dialog_turns', items=[], transformed_items={}, mapped_items={}, transformer_setups={}, mapper_setups={})
    LocalDB(name='dialog_turns', items=[DialogTurn(id='e3b48bcc-df68-43a4-aa81-93922b619293', user_id=None, session_id=None, order=None, user_query='What are the benefits of renewable energy?', assistant_response='I can see you are interested in renewable energy. Renewable energy technologies not only help in reducing greenhouse gas emissions but also contribute significantly to the economy by creating jobs in the manufacturing and installation sectors. The growth in renewable energy usage boosts local economies through increased investment in technology and infrastructure.', user_query_timestamp='2021-09-01T12:00:00Z', assistant_response_timestamp='2021-09-01T12:00:01Z', metadata=None, vector=None), DialogTurn(id='21f0385d-d19a-442f-ae99-910e984cdb65', user_id=None, session_id=None, order=None, user_query='How do solar panels impact the environment?', assistant_response='Solar panels convert sunlight into electricity by allowing photons, or light particles, to knock electrons free from atoms, generating a flow of electricity. Solar panels are a type of renewable energy technology that has been found to have a significant positive effect on the environment by reducing the reliance on fossil fuels.', user_query_timestamp='2021-09-01T12:00:02Z', assistant_response_timestamp='2021-09-01T12:00:03Z', metadata=None, vector=None)], transformed_items={}, mapped_items={}, transformer_setups={}, mapper_setups={})


**Data Processing/Transformation Pipeline(such as TextSplitter and Embedder)**

The `LocalDB` will save different transformations in ``transformed_items`` with either user defined key or the default key ``default``.
Here is how to apply the data transformation to the ``items`` in ``dialog_turn_db``:

.. code-block:: python

    key = "split_and_embed"
    dialog_turn_db.transform_data(data_transformer, map_fn=map_dialogturn_to_document, key=key)
    print(dialog_turn_db.transformed_items[key])

**Save/Reload Data**

.. code-block:: python

    dialog_turn_db.save_state(filepath='.storage/dialog_turns.pkl')
    reloaded_dialog_turn_db = LocalDB.load_state(filepath='.storage/dialog_turns.pkl')
    print(reloaded_dialog_turn_db)

Here is the reloaded_dialog_turn_db

.. code-block::

    LocalDB(name='dialog_turns', items=[DialogTurn(id='72daef1d-5731-427c-b2fd-d738a95bddc7', user_id=None, session_id=None, order=None, user_query='What are the benefits of renewable energy?', assistant_response='I can see you are interested in renewable energy. Renewable energy technologies not only help in reducing greenhouse gas emissions but also contribute significantly to the economy by creating jobs in the manufacturing and installation sectors. The growth in renewable energy usage boosts local economies through increased investment in technology and infrastructure.', user_query_timestamp='2021-09-01T12:00:00Z', assistant_response_timestamp='2021-09-01T12:00:01Z', metadata=None, vector=None), DialogTurn(id='6ab9179f-fa00-4189-b068-91f16f4d9441', user_id=None, session_id=None, order=None, user_query='How do solar panels impact the environment?', assistant_response='Solar panels convert sunlight into electricity by allowing photons, or light particles, to knock electrons free from atoms, generating a flow of electricity. Solar panels are a type of renewable energy technology that has been found to have a significant positive effect on the environment by reducing the reliance on fossil fuels.', user_query_timestamp='2021-09-01T12:00:02Z', assistant_response_timestamp='2021-09-01T12:00:03Z', metadata=None, vector=None)], transformed_items={'split_and_embed': [Document(id=77d1e191-4a20-4b05-b4a7-d7da3b107152, text='What are the benefits of renewable energy? I can see you are interested in renewable energy. Renewab...', meta_data=None, vector='len: 256', parent_doc_id=72daef1d-5731-427c-b2fd-d738a95bddc7, order=0, score=None), Document(id=75eaf430-7b63-4e4c-b3fe-4eba7288c4e3, text='and installation sectors. The growth in renewable energy usage boosts local economies through increa...', meta_data=None, vector='len: 256', parent_doc_id=72daef1d-5731-427c-b2fd-d738a95bddc7, order=1, score=None), Document(id=a1a85d93-92dd-4d39-8b5e-0f29511b186e, text='How do solar panels impact the environment? Solar panels convert sunlight into electricity by allowi...', meta_data=None, vector='len: 256', parent_doc_id=6ab9179f-fa00-4189-b068-91f16f4d9441, order=0, score=None), Document(id=8118c379-e3a3-4076-94fe-07cc64fc42ae, text='has been found to have a significant positive effect on the environment by reducing the reliance on ...', meta_data=None, vector='len: 256', parent_doc_id=6ab9179f-fa00-4189-b068-91f16f4d9441, order=1, score=None)]}, mapped_items={}, transformer_setups={}, mapper_setups={})


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
   - :class:`components.data_process.DocumentSplitter`
   - :class:`components.data_process.ToEmbeddings`
