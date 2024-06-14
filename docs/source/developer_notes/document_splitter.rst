Document Splitter
========================
LLMs’s context window is limited and the performance often drops with very long and nonsense input.
Shorter content is more manageable and fits memory constraint.
The goal of the document splitter is to chunk large data into smaller ones, potentially improving embedding and retrieving.
In this tutorial, we will learn to implement ``LightRAG`` splitters.

TextSplitter
-----------------
The ``TextSplitter`` is designed to efficiently process and chunk **plain text**. 
It leverages configurable separators to facilitate the splitting of :obj:`document object <core.types.Document>` into smaller manageable document chunks.

Capabilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Text Chunking:** Breaks down large text by ``Python str.split()``. 

**Customizability:** Supports customization of separators, allowing developers to define how the text is segmented based on the use case.

**Scalability:** Optimized for performance, handling extensive documents efficiently, with chunk size and overlap control. The raw documents metadata will not be changed.

Key Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``split_by`` specifies the separator by which the document should be split, i.e. the smallest unit during splitting. 
We apply ``Python str.split()`` to break the text into a ``list`` of units. 
Check the following table for ``split_by`` options:

.. list-table:: Text Splitting Options
   :widths: 10 15 75
   :header-rows: 1

   * - ``split_by`` Option
     - Actual Separator
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

.. note::

  The splitted ``list`` will get **concatenated** based on the specified ``chunk_size`` and ``chunk_overlap`` later.

* ``chunk_size`` is the the maximum number of units in each chunk. 

* ``chunk_overlap`` is the number of units that each chunk should overlap. Including context at the borders prevents sudden meaning shift in text between sentences/context, especially in sentiment analysis.

Usage Example
^^^^^^^^^^^^^^^^

.. code-block:: python

    from lightrag.core.document_splitter import TextSplitter
    from lightrag.core.types import Document

    # Configure the splitter settings
    text_splitter = TextSplitter(
        split_by="sentence",
        chunk_size=5,
        chunk_overlap=1
    )

    # Example document
    doc = Document(
        text="Example text. More example text. Even more text to illustrate.",
        id="doc1"
    )

    # Execute the splitting
    splitted_docs = text_splitter.call(documents=[doc])

    for doc in splitted_docs:
        print(doc)

Integration with Other Document Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``TextSplitter`` is basic and mainly supports plain text.
For **PDFs**, developers will need to extract the text before using the splitter. Libraries like ``PyPDF2`` or ``PDFMiner`` can be utilized for this purpose.

``LightRAG``'s future implementations will introduce splitters for ``JSON``, ``HTML``, ``markdown``, and ``code``.

Customization Tips
~~~~~~~~~~~~~~~~~~~~~

To adapt the ``TextSplitter`` for specific separation, developers can modify the ``separators dictionary`` to include unique delimiters to their text formats. This flexibility allows the splitter to function effectively across various domains and document types.
