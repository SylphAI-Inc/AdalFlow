"""Leverage a postgres database to store and retrieve documents."""

from typing import List, Optional, Any
from enum import Enum
import numpy as np
import logging

from lightrag.core.retriever import (
    Retriever,
)
from lightrag.core.embedder import Embedder

from lightrag.core.types import (
    RetrieverOutput,
    RetrieverStrQueryType,
    RetrieverStrQueriesType,
    Document,
)
from lightrag.database.sqlalchemy.sqlachemy_manager import DatabaseManager

log = logging.getLogger(__name__)


class DistanceToOperator(Enum):
    __doc__ = r"""Enum for the distance to operator.

    About pgvector:

    1. L2 distance: <->, inner product (<#>), cosine distance (<=>), and L1 distance (<+>, added in 0.7.0)
    """
    L2 = "<->"
    INNER_PRODUCT = (
        "<#>"  # cosine similarity when the vector is normalized, in range [-1, 1]
    )
    COSINE = "<=>"  # cosine distance, in range [0, 1] = 1 - cosine_similarity
    L1 = "<+>"


class PostgresRetriever(Retriever[Any, RetrieverStrQueryType]):
    __doc__ = r"""Use a postgres database to store and retrieve documents.

    Users can follow this example and to customize the prompt or additionally ask it to output score along with the indices.

    Args:
        top_k (Optional[int], optional): top k documents to fetch. Defaults to 1.
        database_url (str): the database url to connect to. Defaults to postgresql://postgres:password@localhost:5432/vector_db.

    References:
    [1] pgvector extension: https://github.com/pgvector/pgvector
    """

    def __init__(
        self,
        embedder: Embedder,
        top_k: Optional[int] = 1,
        database_url: str = None,
        table_name: str = "document",
        distance_operator: DistanceToOperator = DistanceToOperator.INNER_PRODUCT,
    ):
        super().__init__()
        self.top_k = top_k
        self.table_name = table_name
        db_name = "vector_db"
        self.database_url = (
            database_url or f"postgresql://postgres:password@localhost:5432/{db_name}"
        )
        self.db_manager = DatabaseManager(self.database_url)
        self.embedder = embedder
        self.distance_operator = distance_operator
        self.db_score_prob_fun_map = {
            DistanceToOperator.COSINE: self._convert_cosine_distance_to_probability,
            DistanceToOperator.L2: self._convert_l2_distance_to_probability,
            DistanceToOperator.INNER_PRODUCT: self._convert_cosine_similarity_to_probability,
        }
        self.score_prob_fun = (
            self.db_score_prob_fun_map[self.distance_operator]
            if self.distance_operator in self.db_score_prob_fun_map
            else None
        )

    @classmethod
    def format_vector_search_query(
        cls,
        table_name: str,
        vector_column: str,
        query_embedding: List[float],
        top_k: int,
        distance_operator: DistanceToOperator,
        sort_desc: bool = True,
    ) -> str:
        """
        Formats a SQL query string to select all columns from a table, order the results
        by the distance or similarity score to a provided embedding, and also return
        that score.

        Args:
            table_name (str): The name of the table to query.
            column (str): The name of the column containing the vector data.
            query_embedding (list or str): The embedding vector to compare against.
            top_k (int): The number of top results to return.

        Returns:
            str: A formatted SQL query string that includes the score.
        """

        # Convert the list embedding to a string format suitable for SQL
        if isinstance(query_embedding, list):
            embedding_str = str(query_embedding).replace(
                " ", ""
            )  # Remove spaces for cleaner SQL
        else:
            embedding_str = query_embedding

        # Determine sorting order
        order_by = "DESC" if sort_desc else "ASC"

        # SQL query that includes the score in the selected columns
        sql_query = f"""
        SELECT *, ({vector_column} {distance_operator.value} '{embedding_str}') AS score
        FROM {table_name}
        ORDER BY score {order_by}
        LIMIT {top_k};
        """
        return sql_query

    def retrieve_by_sql(self, query: str) -> List[str]:
        """Retrieve documents from the postgres database."""

        results = self.db_manager.execute_query(query)
        print(results)
        return results

    def _convert_cosine_similarity_to_probability(
        self, cosine_similarity: List[float]
    ) -> List[float]:
        """Convert cosine similarity to probability."""
        return [(1 + cosine_similarity) / 2 for cosine_similarity in cosine_similarity]

    def _convert_l2_distance_to_probability(
        self, l2_distance: List[float]
    ) -> List[float]:
        """Convert L2 distance to probability.

        note:

        Ensure the vector is normalized so that the l2_distance will be in range [0, 2]
        """
        distance = np.array(l2_distance)
        # clip to ensure the distance is in range [0, 2]
        distance = np.clip(distance, 0, 2)
        # convert to probability
        prob_score = 1 - distance / 2
        return prob_score.tolist()

    def _convert_cosine_distance_to_probability(
        self, cosine_distance: List[float]
    ) -> List[float]:
        """Convert cosine distance to probability."""
        return [(1 - cosine_distance) for cosine_distance in cosine_distance]

    def call(
        self,
        input: RetrieverStrQueriesType,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[RetrieverOutput]:
        top_k = top_k or self.top_k
        queries: List[str] = input if isinstance(input, list) else [input]
        retrieved_outputs: List[RetrieverOutput] = []
        queries_embeddings = self.embedder(queries)

        sort_desc = False
        if (
            self.distance_operator == DistanceToOperator.INNER_PRODUCT
        ):  # cosine similarity
            sort_desc = True

        for idx, query in enumerate(queries):
            query_embedding = queries_embeddings.data[idx].embedding
            query_str = self.format_vector_search_query(
                table_name=self.table_name,
                vector_column="vector",
                query_embedding=query_embedding,
                top_k=top_k,
                distance_operator=self.distance_operator,
                sort_desc=sort_desc,
            )
            retrieved_documents = self.retrieve_by_sql(query_str)
            doc_indices = [doc["id"] for doc in retrieved_documents]
            doc_scores = [doc["score"] for doc in retrieved_documents]
            doc_scores_prob = (
                self.score_prob_fun(doc_scores) if self.score_prob_fun else None
            )
            documents: List[Document] = []
            for doc in retrieved_documents:
                documents.append(Document.from_dict(doc))
            retrieved_outputs.append(
                RetrieverOutput(
                    doc_indices=doc_indices,
                    doc_scores=doc_scores_prob if doc_scores_prob else doc_scores,
                    query=query,
                    documents=documents,
                )
            )
        return retrieved_outputs


# if __name__ == "__main__":
#     from lightrag.core.embedder import Embedder
#     from lightrag.core.types import Document
#     from lightrag.database.sqlalchemy.pipeline.default_config import default_config
#     from lightrag.utils import setup_env  # noqa

#     documents = [
#         {
#             "meta_data": {"title": "Li Yin's profile"},
#             "text": "My name is Li Yin, I love rock climbing"
#             + "lots of nonsense text" * 500,
#             "id": "doc1",
#         },
#         {
#             "meta_data": {"title": "Interviewing Li Yin"},
#             "text": "lots of more nonsense text" * 250
#             + "Li Yin is a software developer and AI researcher"
#             + "lots of more nonsense text" * 250,
#             "id": "doc2",
#         },
#     ]
#     db_name = "vector_db"
#     postgres_url = f"postgresql://postgres:password@localhost:5432/{db_name}"

#     vector_config = default_config["to_embeddings"]["component_config"]["embedder"][
#         "component_config"
#     ]
#     eb = Embedder.from_config(vector_config)
#     pr = PostgresRetriever(
#         embedder=eb,
#         database_url=postgres_url,
#         top_k=2,
#         distance_operator=DistanceToOperator.INNER_PRODUCT,
#     )
#     output = pr("What did the author do?")
#     print(output)
