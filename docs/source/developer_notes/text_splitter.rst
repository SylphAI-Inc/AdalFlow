Text Splitter
-----------------
.. admonition:: Author
   :class: highlight

   `Xiaoyi Gu <https://github.com/Alleria1809>`_

In this tutorial, we will learn:

#. Why do we need the ``TextSplitter``

#. How does ``LightRAG's TextSplitter`` work

#. How to implement ``LightRAG's TextSplitter``

Why do we need the ``TextSplitter``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
LLMs’s context window is limited and the performance often drops with very long and nonsense input.
Shorter content is more manageable and fits memory constraint.
The goal of the text splitter is to chunk large data into smaller ones, potentially improving embedding and retrieving.

The ``TextSplitter`` is designed to efficiently process and chunk **plain text**. 
It leverages configurable separators to facilitate the splitting of :obj:`document object <core.types.Document>` into smaller manageable document chunks.

How does ``LightRAG's TextSplitter`` work
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``TextSplitter`` supports 2 types of splitting. 
    
* Type 1: Specify the exact text splitting point such as space<" "> and periods<".">. It is intuitive:
"Hello, world!" -> ["Hello, " ,"world!"]

* Type 2: Use :class:`tokenizer <lightrag.core.tokenizer.Tokenizer>`. It works as:
"Hello, world!" -> ['Hello', ',', ' world', '!']
This aligns with how models see text in the form of tokens. (`Reference <https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb>`_)

Simple text splitting can underestimate the number of tokens. Tokenizer reflects the real token numbers the models take in. 
But the Tokenizer here only works on world level.

* **Overview**:
``TextSplitter`` first utilizes ``split_by`` to specify the text-splitting criterion and breaks the long text into smaller texts.
Then we create a sliding window with length= ``chunk_size``. It moves at step= ``chunk_size`` - ``chunk_overlap``.
The texts inside each window will get concatenated to a smaller chunk. The generated chunks from the splitted text will be returned.

Here are some Definitions:

* **Definitions**
    
``split_by``: Specifies the text-splitting criterion using predefined keys like "word", "sentence", "page", "passage", and "token". The splitter utilizes the corresponding separator from the ``SEPARATORS`` dictionary.

``SEPARATORS``: Maps ``split_by`` criterions to their exact text separators, e.g., spaces<" "> for "word" or periods<"."> for "sentence".

Usage: **SEPARATORS[``split_by``]=separator**

.. note::
    For option ``token``, its separator is "" because we directly split by a tokenizer, instead of text point.

* ``split_by`` specifies the separator by which the document should be split, i.e. the smallest unit during splitting. 
For Type 1 splitting, we apply ``Python str.split()`` to break the text.
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

* ``chunk_size`` is the the maximum number of units in each chunk. 

* ``chunk_overlap`` is the number of units that each chunk should overlap. Including context at the borders prevents sudden meaning shift in text between sentences/context, especially in sentiment analysis.

Here is an example of how ``chunk_size`` works with ``chunk_overlap``:

.. code-block:: python
    from lightrag.core.text_splitter import TextSplitter
    from lightrag.core.types import Document

    # configure the splitter setting
    text_splitter_settings = {
            "split_by": "word",
            "chunk_size": 5,
            "chunk_overlap": 2,
            }

    # set up the document splitter
    text_splitter = TextSplitter(
        split_by=text_splitter_settings["split_by"],
        chunk_size=text_splitter_settings["chunk_size"],
        chunk_overlap=text_splitter_settings["chunk_overlap"],
        )
    doc1 = Document(
    text="Hello, this is lightrag. Please implement your splitter here.",
    id="doc1",
    )

    documents = [doc1]

    splitted_docs = (text_splitter.call(documents=documents))

    for doc in splitted_docs:
        print(doc.text)
    # Output:
    # Hello, this is lightrag. Please 
    # lightrag. Please implement your splitter 
    # your splitter here.
In this case, when splitting by ``word`` with ``chunk_size``=5 and ``chunk_overlap``=2,
each chunk will repeat 2 words from the previous chunk. These 2 words are set by ``chunk_overlap``.
This means each chunk has ``5-2=3`` word(split unit) difference compared with its previous.

.. note::
    ``chunk_overlap`` should always be smaller than ``chunk_size``, otherwise the window won't move and the splitting stucks.


One more example on ``split_by=token``:

.. code-block:: python
    # configure the splitter setting
    text_splitter_settings = {
            "split_by": "token",
            "chunk_size": 5,
            "chunk_overlap": 2,
            }

    # set up the document splitter
    text_splitter = TextSplitter(
        ...
        )

    doc1 = Document(
        text="Hello, this is lightrag. Please implement your splitter here.",
        id="doc1",
        )
    documents = [doc1]
    splitted_docs = (text_splitter.call(documents=documents))
    for doc in splitted_docs:
        print(doc.text)
    # Output:
    # Hello, this is l
    # is lightrag.
    # trag. Please implement your
    # implement your splitter here.
When splitting using tokenizer, each chunk still keeps 5 tokens. 
Since ``lightrag`` -> ['l', 'igh', 'trag'], the second chunk is actually ``is`` + ``l`` + ``igh`` + ``trag`` + ``.``.

.. note::
    The punctuation is considered as a token.

This splitting aligns with how models see text in the form of tokens. (`Reference <https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb>`_)

Simple text splitting(Type 1) can underestimate the number of tokens. Tokenizer reflects the real token numbers the models take in. 
But the Tokenizer here only works at world level.

How to implement ``LightRAG's TextSplitter``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
What you need is to specify the arguments and input your documents this way:

.. code-block:: python

    from lightrag.core.text_splitter import TextSplitter
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
This functionality is ideal for segmenting texts into sentences, words, pages, or passages, which can then be processed further for NLP applications.
For **PDFs**, developers will need to extract the text before using the splitter. Libraries like ``PyPDF2`` or ``PDFMiner`` can be utilized for this purpose.
``LightRAG``'s future implementations will introduce splitters for ``JSON``, ``HTML``, ``markdown``, and ``code``.

Customization Tips
~~~~~~~~~~~~~~~~~~~~~
You can also customize the ``SEPARATORS``. For example, by defining ``SEPARATORS`` = {"question": "?"} and setting ``split_by`` = "question", the document will be split at each ``?``, ideal for processing text structured 
as a series of questions. If you need to customize :class:`tokenizer <lightrag.core.tokenizer.Tokenizer>`, please check `Reference <https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb>`_.
    