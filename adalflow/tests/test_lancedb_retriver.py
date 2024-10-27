import unittest
from unittest.mock import Mock, MagicMock
import numpy as np
from adalflow.components.retriever import LanceDBRetriever
from adalflow.core.embedder import Embedder
from adalflow.core.types import RetrieverOutput, Document

# Mock LanceDB and PyArrow imports since they are specific to LanceDB
lancedb = MagicMock()
pa = MagicMock()

class TestLanceDBRetriever(unittest.TestCase):
    def setUp(self):
        # Basic configuration
        self.dimensions = 128
        self.embedder = Mock(spec=Embedder)
        self.db_uri = "/tmp/test_lancedb"

        # Mock embedding output with a simple structure
        self.dummy_embeddings = np.random.rand(10, self.dimensions).astype(np.float32)
        self.embedder.return_value.data = [
            Mock(embedding=embedding) for embedding in self.dummy_embeddings
        ]

        # Initialize LanceDBRetriever with mocked embedder
        self.retriever = LanceDBRetriever(
            embedder=self.embedder, dimensions=self.dimensions, db_uri=self.db_uri
        )

        # Mock LanceDB table and connection
        self.retriever.db.create_table = MagicMock(return_value=Mock())
        self.retriever.table = self.retriever.db.create_table.return_value

    def test_initialization(self):
        # Check dimensions and embedder assignment
        self.assertEqual(self.retriever.dimensions, self.dimensions)
        self.assertEqual(self.retriever.top_k, 5)

    def test_add_documents(self):
        # Sample documents
        documents = [{"content": f"Document {i}"} for i in range(5)]

        # Mock LanceDB table add method
        self.retriever.table.add = MagicMock()

        # Add documents to LanceDBRetriever
        self.retriever.add_documents(documents)

        # Ensure add method was called
        self.retriever.table.add.assert_called_once()
        # Verify embeddings were passed to LanceDB add method
        added_data = self.retriever.table.add.call_args[0][0]
        self.assertEqual(len(added_data), len(documents))
        self.assertIn("vector", added_data[0])
        self.assertIn("content", added_data[0])

    def test_retrieve(self):
        # Prepare a sample query and mocked search result from LanceDB
        query = "test query"
        dummy_scores = [0.9, 0.8, 0.7]
        dummy_indices = [0, 1, 2]

        # Set up mock search result as if it was retrieved from LanceDB
        self.retriever.table.search = MagicMock(return_value=Mock())
        self.retriever.table.search().limit().to_pandas.return_value = Mock(
            index=dummy_indices, _distance=dummy_scores
        )

        # Retrieve top-k results for the query
        result = self.retriever.retrieve(query)

        # Check if retrieve method returns expected output structure
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], RetrieverOutput)
        self.assertEqual(result[0].query, query)
        self.assertEqual(result[0].doc_indices, dummy_indices)
        self.assertEqual(result[0].doc_scores, dummy_scores)

    def test_retrieve_multiple_queries(self):
        # Prepare multiple queries and mocked search result
        queries = ["query 1", "query 2"]
        dummy_scores = [[0.9, 0.8], [0.85, 0.75]]
        dummy_indices = [[0, 1], [2, 3]]

        # Set up mock for each query's result
        self.retriever.table.search().limit().to_pandas.side_effect = [
            Mock(index=dummy_indices[0], _distance=dummy_scores[0]),
            Mock(index=dummy_indices[1], _distance=dummy_scores[1]),
        ]

        # Retrieve for multiple queries
        results = self.retriever.retrieve(queries)

        # Verify the structure and content of the results
        self.assertEqual(len(results), len(queries))
        for i, result in enumerate(results):
            self.assertEqual(result.query, queries[i])
            self.assertEqual(result.doc_indices, dummy_indices[i])
            self.assertEqual(result.doc_scores, dummy_scores[i])

    def test_empty_document_addition(self):
        # Ensure warning log for empty document list
        with self.assertLogs(level='WARNING'):
            self.retriever.add_documents([])

    def test_retrieve_with_empty_query(self):
        # Check empty query handling, expecting a list with empty RetrieverOutput
        result = self.retriever.retrieve("")
        self.assertEqual(result, [RetrieverOutput(doc_indices=[], doc_scores=[], query="")])

    def test_add_documents_embedding_failure(self):
        # Simulate embedding failure
        self.embedder.side_effect = Exception("Embedding failure")
        documents = [{"content": "test document"}]

        with self.assertRaises(Exception) as context:
            self.retriever.add_documents(documents)

        self.assertEqual(str(context.exception), "Embedding failure")

if __name__ == "__main__":
    unittest.main()
