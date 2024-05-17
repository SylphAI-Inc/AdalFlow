Data classes
============

Compared with the major LLM frameworks, such as LlamaIndex, LangChain, LightRAG has the loosest data structure constraints.

Document
------------
We defined `Document` to function as a `string` container, and it can be used for any kind of text data along its `metadata` and relations
such as `parent_doc_id` if you have ever splitted the documents into chunks, and `embedding` if you have ever computed the embeddings for the document.

It functions as the data input type for some `string`-based components, such as `DocumentSplitter`, `Retriever`.