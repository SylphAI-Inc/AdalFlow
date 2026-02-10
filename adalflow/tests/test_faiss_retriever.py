import unittest
from unittest.mock import Mock
import numpy as np

from adalflow.components.retriever import FAISSRetriever
from adalflow.core.embedder import Embedder, BatchEmbedder
from adalflow.core.functional import normalize_vector
from adalflow.core.types import (
    EmbedderOutput,
    RetrieverOutput,
)


# Helper function to create dummy embeddings
def create_dummy_embeddings(num_embeddings, dim, normalize=True):
    vector = np.random.rand(num_embeddings, dim).astype(np.float32)
    if normalize:
        vector = normalize_vector(vector)
    return vector


class TestFAISSRetriever(unittest.TestCase):

    def setUp(self):
        self.dimensions = 128
        self.num_embeddings = 10
        self.single_query = ["test query"]
        self.embeddings = create_dummy_embeddings(self.num_embeddings, self.dimensions)
        self.embedder = Mock(spec=Embedder)
        self.embedder.return_value = EmbedderOutput(
            data=[Mock(embedding=emb) for emb in self.embeddings][
                0 : len(self.single_query)
            ]
        )
        self.retriever = FAISSRetriever(
            embedder=self.embedder, dimensions=self.dimensions
        )
        self.retriever.build_index_from_documents(self.embeddings)

    def test_initialization(self):
        retriever = FAISSRetriever(
            embedder=self.embedder, top_k=5, dimensions=self.dimensions
        )
        self.assertEqual(retriever.dimensions, self.dimensions)
        self.assertEqual(retriever.top_k, 5)
        self.assertIsNone(retriever.index)

    def test_build_index_from_documents(self):
        self.assertIsNotNone(self.retriever.index)
        self.assertEqual(self.retriever.total_documents, self.num_embeddings)
        self.assertEqual(self.retriever.index.d, self.dimensions)

    def test_retrieve_embedding_queries(self):
        query_embedding = create_dummy_embeddings(1, self.dimensions)
        result = self.retriever.retrieve_embedding_queries(query_embedding)
        self.assertIsInstance(result[0], RetrieverOutput)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].doc_indices), self.retriever.top_k)
        self.assertEqual(len(result[0].doc_scores), self.retriever.top_k)

    def test_retrieve_string_queries(self):
        queries = self.single_query
        result = self.retriever.retrieve_string_queries(queries)
        self.assertIsInstance(result[0], RetrieverOutput)
        self.assertEqual(len(result), len(queries))
        self.assertEqual(len(result[0].doc_indices), self.retriever.top_k)
        self.assertEqual(len(result[0].doc_scores), self.retriever.top_k)

    def test_retrieve_with_empty_index(self):
        empty_retriever = FAISSRetriever(
            embedder=self.embedder, dimensions=self.dimensions
        )
        with self.assertRaises(ValueError):
            empty_retriever.retrieve_string_queries(["test query"])

    def test_cosine_similarity_conversion(self):
        D = np.array([[1, 0, -1]])
        converted = self.retriever._convert_cosine_similarity_to_probability(D)
        expected = np.array([[1, 0.5, 0]])
        np.testing.assert_array_almost_equal(converted, expected, decimal=3)

    def test_reset_index(self):
        self.retriever.reset_index()
        self.assertIsNone(self.retriever.index)
        self.assertEqual(self.retriever.total_documents, 0)

    def test_retrieve_string_queries_with_batch_embedder(self):
        """Test retrieve_string_queries with BatchEmbedder"""
        batch_embedder = Mock(spec=BatchEmbedder)
        # BatchEmbedder returns List[EmbedderOutputType]
        batch_embedder.return_value = [
            EmbedderOutput(
                data=[Mock(embedding=emb) for emb in self.embeddings][
                    0 : len(self.single_query)
                ]
            )
        ]

        retriever = FAISSRetriever(
            embedder=batch_embedder, dimensions=self.dimensions
        )
        retriever.build_index_from_documents(self.embeddings)

        queries = self.single_query
        result = retriever.retrieve_string_queries(queries)
        self.assertIsInstance(result[0], RetrieverOutput)
        self.assertEqual(len(result), len(queries))
        self.assertEqual(len(result[0].doc_indices), retriever.top_k)
        self.assertEqual(len(result[0].doc_scores), retriever.top_k)

    def test_retrieve_non_normalized_embeddings_with_l2_metric(self):
        retriever = FAISSRetriever(
            embedder=self.embedder, dimensions=self.dimensions, metric="euclidean"
        )
        non_normalized_embeddings = create_dummy_embeddings(
            self.num_embeddings, self.dimensions, normalize=False
        )
        retriever.build_index_from_documents(non_normalized_embeddings)

        query_embedding = create_dummy_embeddings(1, self.dimensions, normalize=False)
        result = retriever.retrieve_embedding_queries(query_embedding)
        self.assertIsInstance(result[0], RetrieverOutput)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].doc_indices), retriever.top_k)
        self.assertEqual(len(result[0].doc_scores), retriever.top_k)

    def test_retrieve_normalized_embeddings_with_l2_metric(self):
        retriever = FAISSRetriever(
            embedder=self.embedder, dimensions=self.dimensions, metric="euclidean"
        )
        normalized_embeddings = create_dummy_embeddings(
            self.num_embeddings, self.dimensions, normalize=True
        )
        retriever.build_index_from_documents(normalized_embeddings)

        query_embedding = create_dummy_embeddings(1, self.dimensions, normalize=True)
        result = retriever.retrieve_embedding_queries(query_embedding)
        self.assertIsInstance(result[0], RetrieverOutput)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].doc_indices), retriever.top_k)
        self.assertEqual(len(result[0].doc_scores), retriever.top_k)


if __name__ == "__main__":
    unittest.main()
