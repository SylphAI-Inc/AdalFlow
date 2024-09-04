There are different patterns to build a RAG.

1. RAG with separate data process pipeline and a RAG task pipeline.
   This fits into a scenario where there is lots of data in production database, and we preprocess the data to embeddings and then we build a RAG task pipeline that retrieves context in multiple stages.

2. RAG with dynamic data access. And cache the embedding dynamically in a local storage.
