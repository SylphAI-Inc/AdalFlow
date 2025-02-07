import pytest
from unittest.mock import MagicMock
from adalflow.components.retriever import QdrantRetriever
from adalflow.core.types import RetrieverOutput, Document
from adalflow.core.embedder import Embedder

# Use pytest.importorskip to ensure qdrant_client is installed.
qdrant_client = pytest.importorskip(
    "qdrant_client", reason="qdrant_client not installed"
)

COLLECTION_NAME = "test_collection"


@pytest.fixture
def mock_qdrant_client():
    # Create a MagicMock based on the QdrantClient spec.
    return MagicMock(spec=qdrant_client.QdrantClient)


@pytest.fixture
def qdrant_retriever(mock_qdrant_client):
    """
    Fixture to create a QdrantRetriever with a mock embedder.
    The embedder is set up to return a dummy embedding result with one embedding.
    """
    mock_embedder = MagicMock(spec=Embedder)
    dummy_embedding = MagicMock()
    # Default dummy result: one embedding entry.
    dummy_embedding.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_embedder.return_value = dummy_embedding

    return QdrantRetriever(
        collection_name=COLLECTION_NAME,
        client=mock_qdrant_client,
        embedder=mock_embedder,
        top_k=5,
    )


def test_reset_index(qdrant_retriever, mock_qdrant_client):
    """
    Test that reset_index() calls delete_collection when the collection exists.
    """
    mock_qdrant_client.collection_exists.return_value = True
    qdrant_retriever.reset_index()
    mock_qdrant_client.delete_collection.assert_called_once_with(COLLECTION_NAME)


def test_call_single_query(qdrant_retriever, mock_qdrant_client):
    """
    Test that call() correctly processes a single query.
    Verifies that:
      - The embedder is called with a list containing the query.
      - The output is parsed correctly from the mock query response.
    """
    query = "test query"

    # Set up the embedder to return a dummy embedding result with one item.
    dummy_embedding = MagicMock()
    dummy_embedding.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    qdrant_retriever._embedder.return_value = dummy_embedding

    # Create a mock scored point.
    mock_point = MagicMock()
    mock_point.id = 1
    mock_point.score = 0.9
    mock_point.payload = {"text": "retrieved text", "meta_data": {"key": "value"}}
    mock_point.vector = [0.1, 0.2, 0.3]

    # Prepare a mock query response with a single point.
    mock_query_response = MagicMock()
    mock_query_response.points = [mock_point]

    # Configure the client's query_batch_points to return one response.
    mock_qdrant_client.query_batch_points.return_value = [mock_query_response]

    # Call the retriever with a single query.
    result = qdrant_retriever.call(query)

    # Verify that the embedder was called exactly once with [query].
    assert qdrant_retriever._embedder.call_count == 1
    call_args, _ = qdrant_retriever._embedder.call_args
    assert call_args == ([query],)

    # Verify the returned result.
    assert isinstance(result, list)
    assert len(result) == 1
    output = result[0]
    assert isinstance(output, RetrieverOutput)
    assert output.query == query
    assert output.doc_indices == [1]
    assert output.doc_scores == [0.9]
    assert len(output.documents) == 1
    assert isinstance(output.documents[0], Document)
    assert output.documents[0].text == "retrieved text"
    assert output.documents[0].meta_data == {"key": "value"}


def test_get_first_vector_name(qdrant_retriever, mock_qdrant_client):
    """
    Test _get_first_vector_name() for:
      1. A collection with a non-dict (default, unnamed vector).
      2. A collection with a dict of named vectors.
    """
    # Scenario 1: Default, unnamed vector (non-dict)
    mock_collection = MagicMock()
    mock_collection.config.params.vectors = 42  # not a dict
    mock_qdrant_client.get_collection.return_value = mock_collection

    vector_name = qdrant_retriever._get_first_vector_name()
    assert vector_name is None

    # Scenario 2: Multiple named vectors.
    named_vectors = {"vector1": "details", "vector2": "details"}
    mock_collection.config.params.vectors = named_vectors
    mock_qdrant_client.get_collection.return_value = mock_collection

    vector_name = qdrant_retriever._get_first_vector_name()
    # Expect the first key.
    expected = list(named_vectors.keys())[0]
    assert vector_name == expected


def test_points_to_output():
    """
    Test _points_to_output() to ensure it correctly converts scored points
    into a RetrieverOutput.
    """
    # Create a dummy scored point.
    mock_point = MagicMock()
    mock_point.id = 1
    mock_point.score = 0.9
    mock_point.payload = {"text": "sample text", "meta_data": {"key": "value"}}
    mock_point.vector = [0.1, 0.2, 0.3]

    points = [mock_point]
    query = "test query"
    text_key = "text"
    metadata_key = "meta_data"
    vector_name = "vector_name"

    result = QdrantRetriever._points_to_output(
        points, query, text_key, metadata_key, vector_name
    )

    assert isinstance(result, RetrieverOutput)
    assert result.query == query
    assert result.doc_indices == [1]
    assert result.doc_scores == [0.9]
    assert len(result.documents) == 1
    doc = result.documents[0]
    assert isinstance(doc, Document)
    assert doc.text == "sample text"
    assert doc.meta_data == {"key": "value"}
    assert doc.vector == [0.1, 0.2, 0.3]


def test_doc_from_point():
    """
    Test _doc_from_point() without a vector name.
    """
    mock_point = MagicMock()
    mock_point.id = 1
    mock_point.payload = {"content": "sample text", "some_meta": {"key": "value"}}
    mock_point.vector = [0.1, 0.2, 0.3]

    text_key = "content"
    metadata_key = "some_meta"
    vector_name = None

    document = QdrantRetriever._doc_from_point(
        mock_point, text_key, metadata_key, vector_name
    )

    assert isinstance(document, Document)
    assert document.id == 1
    assert document.text == "sample text"
    assert document.meta_data == {"key": "value"}
    assert document.vector == [0.1, 0.2, 0.3]


def test_doc_from_point_with_vector_name():
    """
    Test _doc_from_point() when the vector is a dict and a vector name is provided.
    """
    mock_point = MagicMock()
    mock_point.id = 1
    mock_point.payload = {"text": "sample text", "meta_data": {"key": "value"}}
    mock_point.vector = {"vector_name": [0.4, 0.5, 0.6]}

    text_key = "text"
    metadata_key = "meta_data"
    vector_name = "vector_name"

    document = QdrantRetriever._doc_from_point(
        mock_point, text_key, metadata_key, vector_name
    )

    assert isinstance(document, Document)
    assert document.id == 1
    assert document.text == "sample text"
    assert document.meta_data == {"key": "value"}
    assert document.vector == [0.4, 0.5, 0.6]


def test_call_with_custom_limit(qdrant_retriever, mock_qdrant_client):
    """
    Test that call() correctly handles a list of queries and uses the custom top_k limit.
    """
    query = "test query"
    custom_limit = 5
    queries = [query, query, query]

    # Prepare dummy embeddings for each query.
    dummy_embeddings = [MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in queries]
    dummy_embedding_result = MagicMock()
    dummy_embedding_result.data = dummy_embeddings
    qdrant_retriever._embedder.return_value = dummy_embedding_result

    # Create a mock scored point.
    mock_point = MagicMock()
    mock_point.id = 1
    mock_point.score = 0.9
    mock_point.payload = {"text": "retrieved text", "meta_data": {"key": "value"}}
    mock_point.vector = [0.1, 0.2, 0.3]

    # Create a mock query response.
    # Note: we use spec from qdrant_client.http.models.QueryResponse if available.
    mock_query_response = MagicMock(
        spec=getattr(qdrant_client.http.models, "QueryResponse", None)
    )
    mock_query_response.points = [mock_point]

    # Set the query_batch_points return value.
    mock_qdrant_client.query_batch_points.return_value = [mock_query_response]

    # Call the retriever with multiple queries.
    result = qdrant_retriever.call(queries, top_k=custom_limit)

    # Verify query_batch_points was called once with the correct collection name.
    mock_qdrant_client.query_batch_points.assert_called_once()
    args, kwargs = mock_qdrant_client.query_batch_points.call_args
    assert args[0] == COLLECTION_NAME

    # Verify that the number of query requests equals the number of input queries.
    requests = kwargs.get("requests")
    assert requests is not None
    assert len(requests) == len(queries)
    for request in requests:
        assert request.limit == custom_limit

    # Optionally, ensure the output is a list.
    assert isinstance(result, list)
