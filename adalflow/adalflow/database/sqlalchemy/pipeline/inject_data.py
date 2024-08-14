from typing import List

from adalflow.core.component import Sequential
from adalflow.core.types import Document

from adalflow.utils.config import new_component
from adalflow.database.sqlalchemy.sqlachemy_manager import DatabaseManager
from adalflow.database.sqlalchemy.model import DocumentModel
from adalflow.database.sqlalchemy.pipeline import default_config


# TODO:  async call
class EmbeddingPipeline:
    __doc__ = r"""Pipeline to process documents and store them in the database

    Args:
        batch_size (int, optional): Batch size for processing. Defaults to 100.

    Example:

    ..code-block:: python

        from adalflow.core.types import Document
        from adalflow.utils import setup_env  # noqa

        documents = [
            {
                "meta_data": {"title": "Li Yin's profile"},
                "text": "My name is Li Yin, I love rock climbing"
                + "lots of nonsense text" * 500,
                "id": "doc1",
            },
            {
                "meta_data": {"title": "Interviewing Li Yin"},
                "text": "lots of more nonsense text" * 250
                + "Li Yin is a software developer and AI researcher"
                + "lots of more nonsense text" * 250,
                "id": "doc2",
            },
        ]
        db_name = "vector_db"
        postgres_url = f"postgresql://postgres:password@localhost:5432/{db_name}"
        ep = EmbeddingPipeline()
        ep.setup_database_manager(postgres_url)

        new_documents = [Document(**doc) for doc in documents]

        ep(new_documents)

    Note:
     - Before you run the pipeline, ensure you have created the database using the `create_tables.py` script
    """

    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size

        self.config_dict = default_config
        self.document_splitter = new_component(self.config_dict["document_splitter"])
        self.to_embeddings = new_component(self.config_dict["to_embeddings"])
        self.vectorizer = self.to_embeddings.vectorizer

        self.data_transformer = Sequential(
            self.document_splitter,
            self.to_embeddings,
        )

    def setup_database_manager(self, db_path: str):
        self.db_manager = DatabaseManager(db_path)

    def process_batch(self, documents: List[Document]):
        transformed_documents = self.data_transformer(documents)
        session = self.db_manager.get_session()
        documents_to_upload = []
        for doc in transformed_documents:
            item = doc.to_dict()
            if "score" in item:
                del item["score"]
            documents_to_upload.append(item)

        DocumentModel.insert_update_bulk(documents_to_upload, session)
        return transformed_documents

    def __call__(self, documents: List[Document]):
        batch_size = self.batch_size
        for i in range(0, len(documents), batch_size):

            List = documents[i : i + batch_size]
            print(i, len(List))
            self.process_batch(List)


if __name__ == "__main__":
    from adalflow.core.types import Document
    from adalflow.utils import setup_env  # noqa

    documents = [
        {
            "meta_data": {"title": "Li Yin's profile"},
            "text": "My name is Li Yin, I love rock climbing"
            + "lots of nonsense text" * 500,
            "id": "doc1",
        },
        {
            "meta_data": {"title": "Interviewing Li Yin"},
            "text": "lots of more nonsense text" * 250
            + "Li Yin is a software developer and AI researcher"
            + "lots of more nonsense text" * 250,
            "id": "doc2",
        },
    ]
    db_name = "vector_db"
    postgres_url = f"postgresql://postgres:password@localhost:5432/{db_name}"
    ep = EmbeddingPipeline()

    ep.setup_database_manager(postgres_url)
    # new_documents = [Document(**doc) for doc in documents]
    # ep(new_documents)

    db_manager = DatabaseManager(postgres_url)
    tables = db_manager.list_table_schemas()
    print(tables)
    column = "vector"
    table_name = "document"
    vectorizer = ep.vectorizer
    query = "What did the author do?"
    query_embedding = vectorizer(query).data[0].embedding

    def format_query_str(
        table_name: str, column: str, query_embedding: List[float], top_k: int = 5
    ):
        return f"""SELECT * FROM {table_name} ORDER BY {column} <-> '{str(query_embedding)}' LIMIT {top_k};"""

    l2_nearest_neighbor = format_query_str(table_name, column, query_embedding)
    print(l2_nearest_neighbor)
    results = db_manager.execute_query(l2_nearest_neighbor)
    print(results)
