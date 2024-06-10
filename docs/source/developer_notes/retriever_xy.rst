Retriever
===================

In this tutorial, we will explain each component in ``LightRAG's Retriever`` and show you how to implement it in your LLM applications.

LLMs develop fast, but they have limitations.

**Content Window Limit:** Although the trend is, LLM models' content window keeps growing, there is still a context limit. 

**Signal to Noise Ratio** Meanwhile, LLMs perform better when the provided contents are relevant to the task.

To improve LLMs performances in production, Retrieval Augmented Generation (RAG), a system that augments LLMs by adding extra context from another source, becomes popular.
**Retrieval**, one of the most important components of RAG, is the process to fetch the extra relevant information to the model.
The common solution for Retrieval is to chunk the documents into smaller contexts, store these pieces in databases such as vectorstore, Graph DB and Relational DB depending on the use case, and create significant embedding representations for these chunks in order to retrieve.

``LightRAG`` aims to find the optimal way to pass the task-requiring data into LLMs.

1. Document Splitter
----------------------

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

2. Embedder
----------------

Now we have splitted long documents to shorter ones, the next part is to retrieve the relevant documents.
But how can we find "relevant" texts? A commonly applied approach in the NLP field is Embedding. 

For ``Embedder`` tutorial, please check `Embedder <./embedder.html>`_.

3. LightRAG Retrievers
------------------------
Given a query, the retriever is responsible to fetch the relevant documents.
Now we have document splitter and embedder, we can check the retrievers now.
LightRAG provides ``FAISSRetriever``, ``InMemoryBM25Retriever``, and ``LLMRetriever``.
These retrievers are built on the basic :class:`Retriever`, with default index building and retrieve phases. 
All these retrievers return a list of ``RetrieverOutput``, including indexes, scores, query and documents. 

#. FAISSRetriever

The ``FAISSRetriever`` uses in-memory Faiss index to retrieve the top k chunks(see `research <https://github.com/facebookresearch/faiss>`_). It is particularly useful in applications involving large-scale vector.
The developers need to configure ``top_k``, ``dimensions`` and ``vectorizer`` first.
``vectorizer`` is basically an instance of the ``Embedder``. The ``FAISSRetriever`` itself will initialize ``faiss.IndexFlatIP`` with the specified ``dimensions`` to do `Exact Search for Inner Product`.

LightRAG's ``FAISSRetriever`` provides :func:`build_index_from_documents <components.retriever.faiss_retriever.FAISSRetriever.build_index_from_documents>` to create index from embeddings(``vector`` field of each document).
It will create ``xb`` indexes(the same number with embeddings). After the indexes are added, the index state will be ``True``.

Then, developers can pass the queries to :func:`retrieve <components.retriever.faiss_retriever.FAISSRetriever.retrieve>`. This function embeds the queries, and performs inner product search for ``xq``(the number of queries) queries and return k most close vectors.
We choose cosine similarity and convert it to range [0, 1] by adding 1 and dividing by 2 to simulate probability. This is how we calculate the score.
Then we attach the score to each retrieval output.

Then, to speed up the retrieval, it is a common practice to build indexes from the documents or chunks.
When the indexes are ready, we should pass the query to the retriever and get the top k documents closest to the query vector.

Here is an example:

.. code-block:: python

    from lightrag.core.embedder import Embedder
    from lightrag.components.model_client import OpenAIClient
    from lightrag.core.data_components import ToEmbedderResponse, ToEmbeddings
    from lightrag.core.types import Document
    from lightrag.core.document_splitter import DocumentSplitter
    from lightrag.components.retriever import FAISSRetriever

    import dotenv
    dotenv.load_dotenv(dotenv_path=".env", override=True)

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # To use ``FAISSRetriever``, we need to prepare the embeddings 
    # for documents or chunks following the previous steps.

    # configure the splitter setting
    text_splitter_settings = {
            "split_by": "word",
            "split_length": 200,
            "split_overlap": 100,
            }

    # set up the document splitter
    text_splitter = DocumentSplitter(
        split_by=text_splitter_settings["split_by"],
        split_length=text_splitter_settings["split_length"],
        split_overlap=text_splitter_settings["split_overlap"],
        )

    doc1 = Document(
        meta_data={"title": "Luna's Profile"},
        text="lots of more nonsense text." * 50
        + "Luna is a domestic shorthair." 
        + "lots of nonsense text." * 100
        + "Luna loves to eat Tuna."
        + "lots of nonsense text." * 50,
        id="doc1",
        )
    doc2 = Document(
        meta_data={"title": "Luna's Hobbies"},
        text="lots of more nonsense text." * 50
        + "Luna loves to eat lickable treats."
        + "lots of more nonsense text." * 50
        + "Luna loves to play cat wand." 
        + "lots of more nonsense text." * 50
        + "Luna likes to sleep all the afternoon",
        id="doc2",
    )
    documents = [doc1, doc2]

    # split the documents
    splitted_docs = (text_splitter.call(documents=documents))

    # configure the vectorizer(embedding) setting
    vectorizer_settings = {
        "model_kwargs": {
            "model": "text-embedding-3-small",
            "dimensions": 256,
            "encoding_format": "float",
        },
        "batch_size": 100
    }

    # set up the embedder using openai model
    vectorizer = Embedder(
            model_client=OpenAIClient,
            model_kwargs=vectorizer_settings["model_kwargs"], # set up model arguments
            output_processors=ToEmbedderResponse(), # convert the model output to EmbedderResponse
        )
    # Prepare embeddings for the documents
    embedder_response_processor = ToEmbeddings(
        vectorizer=vectorizer,
        batch_size=vectorizer_settings["batch_size"],
    )

    # Apply embedding transformation
    embeddings = embedder_response_processor(splitted_docs)

    # Initialize the FAISS retriever with the embeddings
    faiss_retriever = FAISSRetriever(
        top_k=2,
        dimensions=vectorizer_settings["model_kwargs"]["dimensions"],
        vectorizer=vectorizer
    )

    # build indexes for the documents
    faiss_retriever.build_index_from_documents(embeddings) 

    # set up queries
    queries = ["what does luna like to eat?"]

    # get the retrieved results
    faiss_query_result = faiss_retriever.retrieve(query_or_queries=queries)

    # Continue with the rest of your original code
    print(f"*" * 50)
    print("Faiss Retrieval Results:")
    for result in faiss_query_result:
        print(f"Query: {result.query}")
        print(f"Document Indexes: {result.doc_indexes}, Scores: {result.doc_scores}")
        # Fetch and print the document texts corresponding to the retrieved indexes
        for idx in result.doc_indexes:
            print(f"Document ID: {splitted_docs[idx].id} - Title: {splitted_docs[idx].meta_data['title']}")
            print(f"Text: {splitted_docs[idx].text}")  # Print first 200 characters of the document text
            
        print(f"*" * 50)

    # **************************************************
    # Faiss Retrieval Results:
    # Query: what does luna like to eat?
    # Document Indexes: [8 2], Scores: [0.741 0.724]
    # Document ID: e3f04c8b-68ae-4dde-844a-439037e58842 - Title: Luna's Hobbies
    # Text: text. Luna loves to eat lickable treats.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more 
    # Document ID: f2d0f52a-4e69-4cc5-8f78-4499fa22525d - Title: Luna's Profile
    # Text: text.Luna is a domestic shorthair.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots 
    # **************************************************

#. InMemoryBM25Retriever

The ``InMemoryBM25Retriever`` leverages the `Okapi BM25 algorithm(Best Matching 25 ranking) <https://en.wikipedia.org/wiki/Okapi_BM25>`_, a widely-used ranking function in information retrieval that is particularly effective in contexts where document relevance to a query is crucial. 

This retriever is initialized with parameters that fine-tune its behavior:

``top_k``: Number of top documents to retrieve.
``k1``: Controls term frequency saturation.
``b```: Part of the BM25 algorithm that controls the influence of document length on term frequency normalization. Larger b means lengthier documents have more impact on its effect. 0.5 < b < 0.8 is suggested to yields reasonably good results.
``alpha``: Sets a cutoff for the IDF scores, filtering out terms that are too common to be informative.
IDF refers to `Inverse document frequency <https://en.wikipedia.org/wiki/Tf%E2%80%93idf>`_. It measures how much information the word provides.
Lower the IDF score means the word is used a lot and less important in the document.
Please check :class:`InMemoryBM25Retriever` to see how we calculate the IDF score.
``split_function``: Tokenization is customizable via the ``split_function``, which defaults to splitting text by tokens. Here's an example using a custom tokenizer:
The following example shows how the token splitting works. This tokenizer converts text into a series of token IDs, which are numeric representations of the tokens.

.. code-block:: python

    from lightrag.core.tokenizer import Tokenizer
    from typing import List
    def split_text_by_token_fn(tokenizer: Tokenizer, x: str) -> List[str]:
        return tokenizer(x)

    tokenizer = Tokenizer(name="o200k_base")
    sentence =  "Hello world. This is LightRAG."
    print(split_text_by_token_fn(tokenizer=tokenizer, x=sentence))

    # [13225, 2375, 13, 1328, 382, 12936, 49, 2971, 13], these numbers represent token ids

Tokenization can be customized through ``split_function``.

Similar to ``FAISSRetriever``, developers can build index from documents. In ``InMemoryBM25Retriever`` allows direct documents inputs without need for preparing embeddings beforehand.
The ``build_index_from_documents`` first tokenizes the documents, then analyzes each to compute token frequencies necessary for IDF calculation.
And we filter the IDF based on the specified ``alpha``.
The ``t2d`` represents the token and its frequency in documents. 
For example, t2d={"apple":{0:1}} means, the word apple appears once in the 0th document.
With the frequency we can calculate idf. The ``idf`` dictionary is to record the idf score for each token, such as {"apple": 0.9}, it means in the corpus, the token apple has idf score=0.9.

``load_index``, ``save_index`` and ``reset_index`` are supported.


When a query is received, each token of the query is first transformed into its corresponding token using the same ``split_function`` configured during initialization. 

If a token from the query also appears in the documents of the corpus,
the retriever iterates over the documents containing the token, 
applying the BM25 formula to calculate and accumulate scores based on the token's frequency. 
For instance, document 1 = "apple, apple, banana", document 2 = "apple, orange". 
If the query is "apple, orange", the score of document 1 be the accumulated score from 2 "apple". The score of document 2 will be the accumulated score from "apple" and "orange".
The document's score increases for each occurrence of these tokens. 
This cumulative scoring approach ensures that documents containing more query-related tokens are ranked higher. 
Finally, the ``k`` documents with the highest cumulative scores are identified and returned in a ``RetrieverOutput``, 
which means most relevant to the query.

#. LLMRetriever

Unlike ``FAISSRetriever`` and ``InMemoryBM25Retriever``, the ``LLMRetriever`` utilizes LLM models to perform retrieval.

This model-driven approach does not rely on traditional similarity/IDF scores but instead uses the model's understanding of the content.

Besides ``top_k``, developers need to configure the generator arguments to call LLMs, including:
``model_client``: Model provider such as OpenAIClient, or GroqAPIClient.
``model_kwargs``: Model related arguments such the ``temperature``.
``template``: The prompt template used in the generator to guide the model's focus during retrieval.
``preset_prompt_kwargs``: Includes preset arguments for prompt customization, such as ``task_desc_str`` for task descriptions and ``input_str`` for user queries.
``output_processors``: A component by default ``ListParser`` that processes the model's output into a list of document indices. You should configure this parser based on how you instruct the model to output in the prompt.

**Index Building:** When ``build_info_from_documents`` is called, the retriever configures a designed prompt that informs the model of the documents' context. This enables the model to understand and organize the information before any query is processed.
**Retrieve:** Developers can submit queries as a list. The queries will be processed by using the configured model and template.
The retrieve phase will return the k most relevant **document indices** based on the context provided during indexing.
Developers should be aware of the flexibility of prompt instruction and ``output_processors`` setting and process the output indices.

Here is an example for ``LLMRetriever``:

.. code-block:: python

    from lightrag.components.model_client import OpenAIClient
    from lightrag.core.types import Document, RetrieverOutput
    from lightrag.core.document_splitter import DocumentSplitter
    from lightrag.components.retriever import LLMRetriever
    from lightrag.core.string_parser import ListParser

    import dotenv
    dotenv.load_dotenv(dotenv_path=".env", override=True)

    # Document preparation and splitting
    splitter_settings = {"split_by": "word", "split_length": 200, "split_overlap": 100}
    text_splitter = DocumentSplitter(**splitter_settings)
    documents = [
        Document(id="doc1", meta_data={"title": "Luna's Profile"}, text=
                "lots of more nonsense text." * 50
                + "Luna is a domestic shorthair." 
                + "lots of nonsense text." * 50
                + "Luna loves to eat Tuna."
                + "lots of nonsense text." * 50),
        Document(id="doc2", meta_data={"title": "Luna's Hobbies"}, text=
                "lots of more nonsense text." * 50
                + "Luna loves to eat lickable treats."
                + "lots of more nonsense text." * 50
                + "Luna loves to play cat wand." 
                + "lots of more nonsense text." * 50
                + "Luna likes to sleep all the afternoon"),
    ]

    # split the documents
    splitted_docs = text_splitter.call(documents)

    # configure the model
    gpt_model_kwargs = {
            "model": "gpt-4o",
            "temperature": 0.0,
        }
    # set up the retriever
    llm_retriever = LLMRetriever(
        top_k=1,
        model_client=OpenAIClient(),
        model_kwargs=gpt_model_kwargs,
        output_processors = ListParser()
    )

    # build indexes for the splitted documents
    llm_retriever.build_index_from_documents(documents=splitted_docs)

    # set up queries
    queries = ["what does luna like to eat?", "what does Luna look like?"]


    # get the retrieved results indices
    llm_query_indices = llm_retriever.retrieve(query_or_queries=queries)
    # print(llm_query_indices)
    print("*" * 50)
    for query, result in zip(queries, llm_query_indices.data):
        print(f"Query: {query}")
        if result:
            # Retrieve the indices from the result
            document_indices = result
            for idx in document_indices:
                # Ensure the index is within the range of splitted_docs
                if idx < len(splitted_docs):
                    doc = splitted_docs[idx]
                    print(f"Document ID: {doc.id} - Title: {doc.meta_data['title']}")
                    print(f"Text: {doc.text}")  # Print the first 200 characters
                else:
                    print(f"Index {idx} out of range.")
        else:
            print("No documents retrieved for this query.")
        print("*" * 50)

    # **************************************************
    # Query: what does luna like to eat?
    # Document ID: 557cc52b-a2b7-4780-bbc3-f1be8330c167 - Title: Luna's Profile
    # Text: text.Luna is a domestic shorthair.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.Luna loves to eat Tuna.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense 
    # **************************************************
    # Query: what does Luna look like?
    # Document ID: 7de4b00a-e539-4df0-adc9-b4c312bed365 - Title: Luna's Profile
    # Text: text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.lots of more nonsense text.Luna is a domestic shorthair.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense text.lots of nonsense 
    # **************************************************