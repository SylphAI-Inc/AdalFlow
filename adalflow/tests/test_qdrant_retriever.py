import pytest
from unittest.mock import MagicMock
from adalflow.components.retriever import QdrantRetriever
from adalflow.core.types import (
    RetrieverOutput,
    Document,
)
from adalflow.core.embedder import Embedder

qdrant_client = pytest.importorskip(
    "qdrant_client", reason="qdrant_client not installed"
)

COLLECTION_NAME = "test_collection"


@pytest.fixture
def mock_qdrant_client():
    return MagicMock(spec=qdrant_client.QdrantClient)


@pytest.fixture
def qdrant_retriever(mock_qdrant_client):
    return QdrantRetriever(
        collection_name=COLLECTION_NAME,
        client=mock_qdrant_client,
        embedder=MagicMock(spec=Embedder),
        top_k=5,
    )


def test_reset_index(qdrant_retriever, mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = True
    qdrant_retriever.reset_index()
    mock_qdrant_client.delete_collection.assert_called_once_with(COLLECTION_NAME)


def test_call_single_query(qdrant_retriever, mock_qdrant_client):
    query = "test query"

    mock_point = MagicMock()
    mock_point.id = 1
    mock_point.score = 0.9
    mock_point.payload = {"text": "retrieved text", "meta_data": {"key": "value"}}
    mock_point.vector = [0.1, 0.2, 0.3]

    mock_query_response = MagicMock()
    mock_query_response.points = [mock_point]

    mock_qdrant_client.query_batch_points.return_value = [mock_query_response]

    result = qdrant_retriever.call(query)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], RetrieverOutput)
    assert result[0].query == query
    assert len(result[0].doc_indices) == 1
    assert result[0].doc_indices[0] == 1
    assert len(result[0].doc_scores) == 1
    assert result[0].doc_scores[0] == 0.9
    assert len(result[0].documents) == 1
    assert isinstance(result[0].documents[0], Document)
    assert result[0].documents[0].text == "retrieved text"
    assert result[0].documents[0].meta_data == {"key": "value"}


def test_get_first_vector_name(qdrant_retriever, mock_qdrant_client):
    # Check single unnamed vector
    mock_qdrant_client.get_collection.return_value = MagicMock(
        config=MagicMock(
            params=MagicMock(
                vectors=qdrant_client.models.VectorParams(
                    size=1, distance=qdrant_client.models.Distance.COSINE
                )
            )
        )
    )
    vector_name = qdrant_retriever._get_first_vector_name()
    assert vector_name is None

    mock_qdrant_client.get_collection.return_value = MagicMock(
        config=MagicMock(
            params=MagicMock(vectors={"vector1": "details", "vector2": "details"})
        )
    )
    vector_name = qdrant_retriever._get_first_vector_name()
    assert vector_name == "vector1"


def test_points_to_output():
    # Prepare mocked ScoredPoint
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
    assert isinstance(result.documents[0], Document)
    assert result.documents[0].text == "sample text"
    assert result.documents[0].meta_data == {"key": "value"}
    assert result.documents[0].vector == [0.1, 0.2, 0.3]


def test_doc_from_point():
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
    query = "test query"
    custom_limit = 5

    mock_point = MagicMock()
    mock_point.id = 1
    mock_point.score = 0.9
    mock_point.payload = {"text": "retrieved text", "meta_data": {"key": "value"}}
    mock_point.vector = [0.1, 0.2, 0.3]

    mock_query_response = MagicMock(spec=qdrant_client.models.QueryResponse)
    mock_query_response.points = [mock_point]

    mock_qdrant_client.query_batch_points.return_value = [mock_query_response]

    qdrant_retriever.call([query, query, query], top_k=custom_limit)

    mock_qdrant_client.query_batch_points.assert_called_once()

    collection_name = mock_qdrant_client.query_batch_points.call_args[0]
    assert collection_name == (COLLECTION_NAME,)

    requests = mock_qdrant_client.query_batch_points.call_args[1]["requests"]
    for request in requests:
        assert request.limit == custom_limit
