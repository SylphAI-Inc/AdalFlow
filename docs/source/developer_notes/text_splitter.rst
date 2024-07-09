Text Splitter
======================
.. .. admonition:: Author
..    :class: highlight

..    `Xiaoyi Gu <https://github.com/Alleria1809>`_

In this tutorial, we will discuss:

#. TextSplitter Overview

#. How does it work

#. How to use it

#. Chunking Tips

#. Integration with Other Document Types and Customization Tips

TextSplitter Overview
-----------------------------
LLMs’s context window is limited and the performance often drops with very long and nonsense input.
Shorter content is more manageable and fits memory constraint.
The goal of the ``TextSplitter`` is to chunk large data into smaller ones, potentially improving embedding and retrieving.

The ``TextSplitter`` is designed to efficiently process and chunk **plain text**.
It leverages configurable separators to facilitate the splitting of :obj:`document object <core.types.Document>` into smaller manageable document chunks.

How does it work
-----------------------------
``TextSplitter`` first utilizes ``split_by`` to specify the text-splitting criterion and breaks the long text into smaller texts.
Then we create a sliding window with ``length= chunk_size``. It moves at ``step= chunk_size - chunk_overlap``.
The texts inside each window will get merged to a smaller chunk. The generated chunks from the splitted text will be returned.

Splitting Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^
``TextSplitter`` supports 2 types of splitting.
Here are sample examples and you will see the real output of ``TextSplitter`` in the usage section.

* **Type 1:** Specify the exact text splitting point such as space<" "> and periods<".">. E.g. if you set ``split_by = "word"``, you will get:

::

    "Hello, world!" -> ["Hello, " ,"world!"]

* **Type 2:** Use :class:`core.tokenizer.Tokenizer`. It works as:

::

    "Hello, world!" -> ["Hello", ",", " world", "!"]

Tokenization aligns with how models see text in the form of tokens (`Reference <https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb>`_),

.. note::

    Tokenizer reflects the real token numbers the models take in and helps the developers control budgets.

Definitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **split_by** specifies the split rule, i.e. the smallest unit during splitting. We support ``"word"``, ``"sentence"``, ``"page"``, ``"passage"``, and ``"token"``. The splitter utilizes the corresponding separator from the ``SEPARATORS`` dictionary.
For Type 1 splitting, we apply ``Python str.split()`` to break the text.

* **SEPARATORS**: Maps ``split_by`` criterions to their exact text separators, e.g., spaces <" "> for "word" or periods <"."> for "sentence".

.. note::
    For option ``token``, its separator is "" because we directly split by a tokenizer, instead of specific text point.

* **chunk_size** is the the maximum number of units in each chunk. To figure out which ``chunk_size`` works best for you, you can firstly preprocess your raw data, select a range of the ``chunk_size`` and then run the evaluation on your use case with a bunch of queries.

* **chunk_overlap** is the number of units that each chunk should overlap. Including context at the borders prevents sudden meaning shift in text between sentences/context, especially in sentiment analysis.

Here are examples of how ``split_by``, ``chunk_size`` works with ``chunk_overlap``.

Input Document Text:

::

    Hello, this is lightrag. Please implement your splitter here.


.. list-table:: Chunking Example Detailed
   :widths: 15 15 15 55
   :header-rows: 1

   * - Split By
     - Chunk Size
     - Chunk Overlap
     - Resulting Chunks
   * - word
     - 5
     - 2
     - "Hello, this is lightrag. Please", "lightrag. Please implement your splitter", "your splitter here."
   * - sentence
     - 1
     - 0
     - "Hello, this is lightrag.", "Please implement your splitter here."
   * - token
     - 5
     - 2
     - "Hello, this is l", "is lightrag.", "trag. Please implement your", "implement your splitter here."

When splitting by ``word`` with ``chunk_size = 5`` and ``chunk_overlap = 2``,
each chunk will repeat 2 words from the previous chunk. These 2 words are set by ``chunk_overlap``.
This means each chunk has ``5-2=3`` word(split unit) difference compared with its previous.

When splitting using tokenizer, each chunk still keeps 5 tokens.
For example, the tokenizer transforms ``lightrag`` to ['l', 'igh', 'trag']. So the second chunk is actually ``is`` + ``l`` + ``igh`` + ``trag`` + ``.``.

.. note::
    ``chunk_overlap`` should always be smaller than ``chunk_size``, otherwise the window won't move and the splitting stucks.
    Our default tokenization model is ``cl100k_base``. If you use tokenization (``split_by`` = ``token``), the punctuations are also considered as tokens.

How to use it
-----------------------------
What you need is to specify the arguments and input your documents this way:

.. code-block:: python

    from lightrag.components.data_process.text_splitter import TextSplitter
    from lightrag.core.types import Document

    # Configure the splitter settings
    text_splitter = TextSplitter(
        split_by="word",
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

    # Output:
    # Document(id=44a8aa37-0d16-40f0-9ca4-2e25ae5336c8, text='Example text. More example text. ', meta_data=None, vector=[], parent_doc_id=doc1, order=0, score=None)
    # Document(id=ca0af45b-4f88-49b5-97db-163da9868ea4, text='text. Even more text to ', meta_data=None, vector=[], parent_doc_id=doc1, order=1, score=None)
    # Document(id=e7b617b2-3927-4248-afce-ec0fc247ac8b, text='to illustrate.', meta_data=None, vector=[], parent_doc_id=doc1, order=2, score=None)

Chunking Tips
-----------------------------
Choosing the proper chunking strategy involves considering several key factors:

- **Content Type**: Adapt your chunking approach to matching the specific type of content, such as articles, books, social media posts, or genetic sequences.
- **Embedding Model**: Select a chunking method that aligns with your embedding model's training to optimize performance. For example, sentence-based splitting pairs well with `sentence-transformer <https://huggingface.co/sentence-transformers>`_ models, while token-based splitting is ideal for OpenAI's `text-embedding-ada-002 <https://openai.com/index/new-and-improved-embedding-model>`_.
- **Query Dynamics**: The length and complexity of queries should influence your chunking strategy. Larger chunks may be better for shorter queries lacking detailed specifications and needing broad context, whereas longer queries(more specific) might have higher accuracy with finer granularity.
- **Application of Results**: The application, whether it be semantic search, question answering, or summarization, dictates the appropriate chunking method, especially considering the limitations of content windows in large language models (LLMs).
- **System Integration**: Efficient chunking aligns with system capabilities. For example, `Full-Text Search:` Use larger chunks to allow algorithms to explore broader contexts effectively. For example, search books based on extensive excerpts or chapters. `Granular Search Systems:` Employ smaller chunks to precisely retrieve information relevant to user queries, such as retrieving specific instructions directly in response to a user’s question. For example, if a user asks, "How do I reset my password?". The system can retrieve a specific sentence or paragraph addressing that action directly.

   
Chunking Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed-Size Chunking
""""""""""""""""""""""""""

- Ideal for content requiring uniform chunk sizes like genetic sequences or standardized data entries. This method, which involves splitting text into equal-sized word blocks, is simple and efficient but may compromise semantic coherence and risk breaking important contextual links.

Content-Aware Chunking
""""""""""""""""""""""""""

- **Split by Sentence**: Proper for texts needing a deep understanding of complete sentences, such as academic articles or medical reports. This method maintains grammatical integrity and contextual flow.
- **Split by Passage**: Useful for maintaining the structure and coherence of large documents. Supports detailed tasks like question answering and summarization by focusing on specific text sections.
- **Split by Page**: Effective for large documents where each page contains distinct information, such as legal or academic texts, facilitating precise navigation and information extraction.

Token-Based Splitting
""""""""""""""""""""""""""

- Beneficial for scenarios where embedding models have strict token limitations. This method divides text based on token count, optimizing compatibility with LLMs like GPT, though it may slow down processing due to model complexities.

Upcoming Splitting Features
""""""""""""""""""""""""""""""""

- **Semantic Splitting**: Focuses on grouping texts by meaning rather than structure, enhancing the relevance for thematic searches or advanced contextual retrieval tasks.

Integration with Other Document Types
----------------------------------------------------------
This functionality is ideal for segmenting texts into sentences, words, pages, or passages, which can then be processed further for NLP applications.
For **PDFs**, developers will need to extract the text before using the splitter. Libraries like ``PyPDF2`` or ``PDFMiner`` can be utilized for this purpose.
``LightRAG``'s future implementations will introduce splitters for ``JSON``, ``HTML``, ``markdown``, and ``code``.

Customization Tips
-----------------------------
You can also customize the ``SEPARATORS``. For example, by defining ``SEPARATORS`` = ``{"question": "?"} ``and setting ``split_by = "question"``, the document will be split at each ``?``, ideal for processing text structured
as a series of questions. If you need to customize :class:`core.tokenizer.Tokenizer`, please check `Reference <https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb>`_.
