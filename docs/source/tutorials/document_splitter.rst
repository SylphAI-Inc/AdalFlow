Retriever
===================

In this tutorial, we will explain each component in ``LightRAG's Retriever`` and show you how to implement it in your LLM applications.

LLMs develop fast, but they have limitations.

**Content Window Limit:** Although the trend is, LLM models' content window keeps growing, there is still a context limit. 

**Signal to Noise Ratio** Meanwhile, LLMs perform better when the provided contents are relavant to the task.

To improve LLMs performances in production, Retrieval Augmented Generation (RAG), a system that augments LLMs by adding extra context from another source, becomes popular.
**Retrieval**, one of the most important components of RAG, is the process to fetch the extra relavant information to the model.
The common solution for Retrieval is to chunk the documents into smaller contexts, store these pieces in databases such as vectorstore, Graph DB and Relational DB depending on the use case, and create embedding for these chunks in order to retrieve.
Besides RAG, Retrieval can be used in simpler use case such as few shot example Retrieval.

``LightRAG`` aims to find the optimal way to pass the task-requiring data into LLMs.

1. Document Splitter
----------------------

``LightRAG's DocumentSplitter`` splits a list of documents(:class:`core.data_classes.Document`) into a list of text documents with shorter texts. 
Each :class:`core.data_classes.Document` object is a text container with optional metadata and vector representation.
We use document object to manage id, document content, meta data, document's embedding vectors, etc.
Instead of maintaining the complex relationship between parent, child, previous, and next documents, ``LightRAG`` mainly use ``parent_doc_id``(id of the Document where the chunk is from) and ``order``(order of the chunked document in the original document).

It's easy to implement the ``DocumentSplitter``.
First, you should import the necessary components.

.. code:: python

    from core.document_splitter import DocumentSplitter
    from core.data_classes import Document

Then, configure the splitter settings. 
Here ``split_by`` is the unit by which the document should be split. Options are ``word``, ``sentence``, ``page``, ``passage``. We will use ``word`` as an example.
``split_length`` is the the maximum number of units in each split. 
``split_overlap`` is the number of units that each split should overlap. With the overlap, each chunk will keep several context words and make sense to the LLM. You will see how it works in the example.

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
For example, the data chunks for example 3 will be: 

1. Review: What a fantastic movie! Had a great time and would watch it again! Sentiment:,
2. again! Sentiment: Positive

The first sentence has 15 words. The second sentence has 3 words, the first 2 words are overlapped with the previous sentence.

.. code:: python

    splitted_docs = (text_splitter.call(documents=documents))

    # output:
    # splitted_doc: [Document(id=15d838c4-abda-4c39-b81f-9cd745effb43, meta_data=None, text=Review: I absolutely loved the friendly staff and the welcoming atmosphere! Sentiment: Positive, estimated_num_tokens=17), Document(id=e4850140-8762-4972-9bae-1dfe96ccb65f, meta_data=None, text=Review: It was an awful experience, the food was bland and overpriced. Sentiment: Negative, estimated_num_tokens=21), Document(id=6bd772b9-88b4-4dfa-a595-922c0f8a4efb, meta_data=None, text=Review: What a fantastic movie! Had a great time and would watch it again! Sentiment: , estimated_num_tokens=21), Document(id=b0d98c1b-13ac-4c92-882e-2ed0196b0c81, meta_data=None, text=again! Sentiment: Positive, estimated_num_tokens=6), Document(id=fdc2429b-17e7-4c00-991f-f89e0955e3a3, meta_data=None, text=Review: The store is not clean and smells bad. Sentiment: Negative, estimated_num_tokens=15)]

