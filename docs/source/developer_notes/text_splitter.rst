TextSplitter
===================
.. admonition:: Author
   :class: highlight

   `Xiaoyi Gu <https://github.com/Alleria1809>`_

The DocumentSplitter in LightRAG is designed to preprocess text by splitting long documents into smaller chunks. 
This improves the performance of embedding models and ensures they operate within their maximum context length limits. 

``LightRAG's DocumentSplitter`` splits a list of documents (:obj:`core.base_data_class.Document`) into a list of shorter documents.
The document object to manage id, document content,optional meta data, document's embedding vectors, etc.
Instead of maintaining the complex relationship between parent, child, previous, and next documents, ``LightRAG`` mainly manages the related documents with ``parent_doc_id`` (id of the Document where the chunk is from) and ``order`` (order of the chunked document in the original document).

**Key Arguments:**

* ``split_by`` is the unit by which the document should be split. We implemented a string split function inside to break the text into a ``list``. The splitted ``list`` will get concatenated based on the specified ``split_length`` later.
Check the following table for ``split_by`` options:

.. list-table:: Text Splitting Options
   :widths: 10 15 75
   :header-rows: 1

   * - Option
     - Split by
     - Example
   * - **page**
     - ``\f``
     - ``Hello, world!\fNew page starts here.`` to ``['Hello, world!\x0c', 'New page starts here.']``
   * - **passage**
     - ``\n\n``
     - ``Hello, world!\n\nNew paragraph starts here`` to ``['Hello, world!\n\n', 'New paragraph starts here.']``
   * - **sentence**
     - ``.``
     - ``Hello, world. This is LightRAG.`` to ``['Hello, world.', ' This is LightRAG.', '']``
   * - **word**
     - ``<space>``
     - ``Hello, world. This is LightRAG.`` to ``['Hello, ', 'world. ', 'This ', 'is ', 'LightRAG.']``

We will use ``word`` in our example. 

* ``split_length`` is the the maximum number of units in each split. 

* ``split_overlap`` is the number of units that each split should overlap. Including context at the borders prevents sudden meaning shift in text between sentences/context, especially in sentiment analysis. In ``LightRAG`` we use ``windowed`` function in ``more-itertools`` package to build a sliding window for the texts to keep the overlaps. The window step size = ``split_length - split_overlap``.

After splitting the long text into a list and using a sliding window to generate the text lists with specified overlap length, the text list will be concatenated into text pieces again.
Here is a quick example:

``Review: The theater service is terrible. The movie is good.`` Set ``split_by: word``, ``split_length: 6``, ``split_overlap: 2``. 

With our ``DocumentSplitter`` logic, the output will be: ``Review: The theater service is terrible.``, ``is terrible. The movie is good.``
It prevents the model of misunderstand the context. If we don't have overlap, the second sentence will be ``The movie is good.`` and the embedding model might only consider this document is merely ``Positive``.

Now let's see the code example. First, import the components.

.. code:: python

    from core.document_splitter import DocumentSplitter
    from core.base_data_class import Document

Then, configure the splitter settings.

.. code:: python

    text_splitter_settings = {
        "split_by": "word",
        "split_length": 15,
        "split_overlap": 2,
        }

Next, define the document splitter and set up the documents.

.. code:: python

    text_splitter = DocumentSplitter(
    split_by=text_splitter_settings["split_by"],
    split_length=text_splitter_settings["split_length"],
    split_overlap=text_splitter_settings["split_overlap"],
    )

    example1 = Document(
        text="Review: I absolutely loved the friendly staff and the welcoming atmosphere! Sentiment: Positive",
    )
    example2 = Document(
        text="Review: It was an awful experience, the food was bland and overpriced. Sentiment: Negative",
    )
    example3 = Document(
        text="Review: What a fantastic movie! Had a great time and would watch it again! Sentiment: Positive",
    )
    example4 = Document(
        text="Review: The store is not clean and smells bad. Sentiment: Negative",
    )

    documents = [example1, example2, example3, example4]

Now you can use the splitter to create document chunks.

.. code:: python

    splitted_docs = (text_splitter.call(documents=documents))

    # output:
    # splitted_doc: [Document(id=15d838c4-abda-4c39-b81f-9cd745effb43, meta_data=None, text=Review: I absolutely loved the friendly staff and the welcoming atmosphere! Sentiment: Positive, estimated_num_tokens=17), Document(id=e4850140-8762-4972-9bae-1dfe96ccb65f, meta_data=None, text=Review: It was an awful experience, the food was bland and overpriced. Sentiment: Negative, estimated_num_tokens=21), Document(id=6bd772b9-88b4-4dfa-a595-922c0f8a4efb, meta_data=None, text=Review: What a fantastic movie! Had a great time and would watch it again! Sentiment: , estimated_num_tokens=21), Document(id=b0d98c1b-13ac-4c92-882e-2ed0196b0c81, meta_data=None, text=again! Sentiment: Positive, estimated_num_tokens=6), Document(id=fdc2429b-17e7-4c00-991f-f89e0955e3a3, meta_data=None, text=Review: The store is not clean and smells bad. Sentiment: Negative, estimated_num_tokens=15)]