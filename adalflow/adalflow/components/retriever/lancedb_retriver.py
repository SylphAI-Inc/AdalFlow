import logging
import numpy as np
import pyarrow as pa
import lancedb
from typing import List, Optional, Sequence, Union, Dict, Any
from adalflow.core.embedder import Embedder
from adalflow.core.types import ModelClientType, RetrieverOutput, RetrieverOutputType

# Initialize logging
log = logging.getLogger(__name__)

# Defined data types
LanceDBRetrieverDocumentEmbeddingType = Union[List[float], np.ndarray]  # single embedding
LanceDBRetrieverDocumentsType = Sequence[LanceDBRetrieverDocumentEmbeddingType]

# Step 2: Define the LanceDBRetriever class
class LanceDBRetriever:
    def __init__(self, embedder: Embedder, dimensions: int, db_uri: str = "/tmp/lancedb", top_k: int = 5, overwrite: bool = True):
        self.db = lancedb.connect(db_uri)
        self.embedder = embedder
        self.top_k = top_k
        self.dimensions = dimensions

        # Define table schema with vector field for embeddings
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), list_size=self.dimensions)),
            pa.field("content", pa.string())
        ])

        # Create or overwrite the table for storing documents and embeddings
        self.table = self.db.create_table("documents", schema=schema, mode="overwrite" if overwrite else "append")

    def add_documents(self, documents: Sequence[Dict[str, Any]]):
        """Adds documents with pre-computed embeddings."""
        if not documents:
            log.warning("No documents provided for embedding")
            return

        # Embed document content using Embedder
        doc_texts = [doc["content"] for doc in documents]
        embeddings = self.embedder(input=doc_texts).data

        # Format embeddings for LanceDB
        data = [{"vector": embedding.embedding, "content": text} for embedding, text in zip(embeddings, doc_texts)]

        # Add data to LanceDB table
        self.table.add(data)
        log.info(f"Added {len(documents)} documents to the index")

    def retrieve(self, query: Union[str, List[str]], top_k: Optional[int] = None) -> List[RetrieverOutput]:
        """Retrieve top-k documents from LanceDB for given query or queries."""
        if isinstance(query, str):
            query = [query]

        # Embed the query text(s) with Embedder
        query_embeddings = self.embedder(input=query).data
        output: List[RetrieverOutput] = []

        # Perform search in LanceDB for each query
        for query_emb in query_embeddings:
            results = (
                self.table.search(query_emb.embedding)
                .limit(top_k or self.top_k)
                .to_pandas()
            )

            # Gather indices and scores from search results
            indices = results.index.tolist()
            scores = results["_distance"].tolist()

            # Append results to output
            output.append(RetrieverOutput(
                doc_indices=indices,
                doc_scores=scores,
                query=query[0] if len(query) == 1 else query
            ))
        return output
