import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from adalflow.components.retriever import LanceDBRetriever
from adalflow.core.embedder import Embedder
from unittest import mock
from adalflow.core.types import EmbedderOutput, RetrieverOutput

# Helper function to create dummy embeddings
def create_dummy_embeddings(num_embeddings, dim):
    return np.random.rand(num_embeddings, dim).astype(np.float32)

class TestLanceDBRetriever(unittest.TestCase):
    def setUp(self):
        self.dimensions = 128
        self.top_k = 5
        self.single_query = ["sample query"]
        self.embedder = Mock(spec=Embedder)

        # Mock embedder to return dummy embeddings
        self.dummy_embeddings = create_dummy_embeddings(10, self.dimensions)
        self.embedder.return_value = EmbedderOutput(
            data=[Mock(embedding=emb) for emb in self.dummy_embeddings[:len(self.single_query)]]
        )

        with patch("lancedb.connect") as mock_db_connect:
            self.mock_db = mock_db_connect.return_value
            self.mock_table = Mock()
            self.mock_db.create_table.return_value = self.mock_table
            self.retriever = LanceDBRetriever(
                embedder=self.embedder,
                dimensions=self.dimensions,
                db_uri="/tmp/lancedb",
                top_k=self.top_k
            )

    def test_initialization(self):
        self.assertEqual(self.retriever.dimensions, self.dimensions)
        self.assertEqual(self.retriever.top_k, self.top_k)
        self.mock_db.create_table.assert_called_once()

    def test_add_documents(self):
        documents = [{"content": f"Document {i}"} for i in range(10)]
        embeddings = create_dummy_embeddings(len(documents), self.dimensions)

        # Mock embedding output
        self.embedder.return_value = EmbedderOutput(
            data=[Mock(embedding=embedding) for embedding in embeddings]
        )

        self.retriever.add_documents(documents)
        self.assertEqual(self.mock_table.add.call_count, 1)
        args, _ = self.mock_table.add.call_args
        self.assertEqual(len(args[0]), len(documents))

    def test_add_documents_no_documents(self):
        self.retriever.add_documents([])
        self.mock_table.add.assert_not_called()

    def test_retrieve_single_query(self):
        query = "sample query"
        query_embedding = create_dummy_embeddings(1, self.dimensions)[0]

        # Mock embedding for query
        self.embedder.return_value = EmbedderOutput(
            data=[Mock(embedding=query_embedding)]
        )

        # Mock search results from LanceDB as pandas DataFrame
        results_df = pd.DataFrame({
            "index": [0, 1, 2],
            "_distance": [0.1, 0.2, 0.3]
        })
        self.mock_table.search.return_value.limit.return_value.to_pandas.return_value = results_df

        result = self.retriever.retrieve(query)
        self.assertIsInstance(result[0], RetrieverOutput)
        self.assertEqual(len(result[0].doc_indices), 3)
        self.assertEqual(len(result[0].doc_scores), 3)
        self.assertListEqual(result[0].doc_indices, [0, 1, 2])
        self.assertListEqual(result[0].doc_scores, [0.1, 0.2, 0.3])

    def test_retrieve_multiple_queries(self):
        queries = ["query 1", "query 2"]
        query_embeddings = create_dummy_embeddings(len(queries), self.dimensions)

        # Mock embedding for queries
        self.embedder.return_value = EmbedderOutput(
            data=[Mock(embedding=embedding) for embedding in query_embeddings]
        )

        # Mock search results for each query
        results_df = pd.DataFrame({
            "index": [0, 1, 2],
            "_distance": [0.1, 0.2, 0.3]
        })
        self.mock_table.search.return_value.limit.return_value.to_pandas.return_value = results_df

        result = self.retriever.retrieve(queries)
        self.assertEqual(len(result), len(queries))
        for res in result:
            self.assertIsInstance(res, RetrieverOutput)
            self.assertEqual(len(res.doc_indices), 3)
            self.assertEqual(len(res.doc_scores), 3)

    def test_retrieve_with_empty_query(self):
        # Mock the empty results DataFrame
        self.mock_table.search.return_value.limit.return_value.to_pandas.return_value = pd.DataFrame({
            "index": [],
            "_distance": []
        })

    def test_retrieve_with_no_index(self):
        empty_retriever = LanceDBRetriever(
            embedder=self.embedder, dimensions=self.dimensions
        )
        with self.assertRaises(ValueError):
            empty_retriever.retrieve("test query")

    def test_overwrite_table_on_initialization(self):
        with patch("lancedb.connect") as mock_db_connect:
            mock_db = mock_db_connect.return_value
            mock_table = Mock()
            mock_db.create_table.return_value = mock_table

            LanceDBRetriever(
                embedder=self.embedder,
                dimensions=self.dimensions,
                db_uri="/tmp/lancedb",
                overwrite=True
            )
            mock_db.create_table.assert_called_once_with(
                "documents",
                schema=mock.ANY,
                mode="overwrite"
            )


if __name__ == "__main__":
    unittest.main()
