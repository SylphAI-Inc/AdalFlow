import logging
import pyarrow as pa
import lancedb
from typing import List, Optional, Sequence, Union, Dict, Any

from adalflow.core.embedder import Embedder
from adalflow.core.retriever import Retriever
from adalflow.core.types import RetrieverOutput

from adalflow.core.types import (
    RetrieverStrQueryType,
)

log = logging.getLogger(__name__)


class LanceDBRetriever(Retriever[Any, RetrieverStrQueryType]):

    __doc__ = r"""
        LanceDBRetriever is a retriever that leverages LanceDB to efficiently store and query document embeddings.

        Args:
            embedder (Embedder): An instance of the Embedder class used for computing embeddings.
            dimensions (int): The dimensionality of the embeddings used.
            db_uri (str): The URI of the LanceDB storage (default is "/tmp/lancedb").
            top_k (int): The number of top results to retrieve for a given query (default is 5).
            overwrite (bool): If True, the existing table is overwritten; otherwise, new documents are appended.

        This retriever supports adding documents with their embeddings to a LanceDB storage and retrieving relevant documents based on a given query.

        More information on LanceDB can be found here:(https://github.com/lancedb/lancedb)
        Documentations: https://lancedb.github.io/lancedb/
        """

    def __init__(
        self,
        embedder: Embedder,
        dimensions: int,
        db_uri: str = "/tmp/lancedb",
        top_k: int = 5,
        overwrite: bool = True,
    ):

        super().__init__()
        self.db = lancedb.connect(db_uri)
        self.embedder = embedder
        self.top_k = top_k
        self.dimensions = dimensions

        schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), list_size=self.dimensions)),
                pa.field("content", pa.string()),
            ]
        )

        self.table = self.db.create_table(
            "documents", schema=schema, mode="overwrite" if overwrite else "append"
        )

    def add_documents(self, documents: Sequence[Dict[str, Any]]):
        """
        Adds documents with and computes their embeddings using the provided Embedder.
        Args:
            documents (Sequence[Dict[str, Any]]): A sequence of documents, each with a 'content' field containing text.

        """
        if not documents:
            log.warning("No documents provided for embedding")
            return

        # Embed document content using Embedder
        doc_texts = [doc["content"] for doc in documents]
        embeddings = self.embedder(input=doc_texts).data

        # Format embeddings for LanceDB
        data = [
            {"vector": embedding.embedding, "content": text}
            for embedding, text in zip(embeddings, doc_texts)
        ]

        self.table.add(data)
        log.info(f"Added {len(documents)} documents to the index")

    def retrieve(
        self, query: Union[str, List[str]], top_k: Optional[int] = None
    ) -> List[RetrieverOutput]:
        """.
        Retrieve top-k documents from LanceDB for a given query or queries.
        Args:
            query (Union[str, List[str]]): A query string or a list of query strings.
            top_k (Optional[int]): The number of top documents to retrieve (if not specified, defaults to the instance's top_k).

        Returns:
            List[RetrieverOutput]: A list of RetrieverOutput containing the indices and scores of the retrieved documents.
        """
        if isinstance(query, str):
            query = [query]

        if not query or (isinstance(query, str) and query.strip() == ""):
            raise ValueError("Query cannot be empty.")

        if not self.table:
            raise ValueError(
                "The index has not been initialized or the table is missing."
            )

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
            output.append(
                RetrieverOutput(
                    doc_indices=indices,
                    doc_scores=scores,
                    query=query[0] if len(query) == 1 else query,
                )
            )
        return output
