from typing import List
import os

from lightrag.core.component import Sequential
from lightrag.core.types import Document

from lightrag.utils.config import new_component
from lightrag.utils.file_io import load_json
from lightrag.database.sqlalchemy.sqlachemy_manager import DatabaseManager
from lightrag.database.sqlalchemy.model import DocumentModel


# TODO: async call
class EmbeddingPipeline:
    __doc__ = r"""Pipeline to process documents and store them in the database

    Args:
        batch_size (int, optional): Batch size for processing. Defaults to 100.

    Example:

    ..code-block:: python

        from lightrag.core.types import Document
        from lightrag.utils import setup_env  # noqa

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
        current_script_dir = os.path.dirname(os.path.realpath(__file__))

        config_path = f"{current_script_dir}/default_config.json"
        print(config_path)

        self.config_dict = load_json(config_path)
        print(self.config_dict)
        self.document_splitter = new_component(self.config_dict["document_splitter"])
        self.to_embeddings = new_component(self.config_dict["to_embeddings"])

        self.data_transformer = Sequential(
            self.document_splitter,
            self.to_embeddings,
        )

    def setup_database_manager(self, db_path: str):
        self.db_manager = DatabaseManager(db_path)

    def process_batch(self, documents: List[Document]):
        transformed_documents = self.data_transformer(documents)
        session = self.db_manager.get_session()
        DocumentModel.insert_update_bulk(
            [doc.to_dict() for doc in transformed_documents], session
        )
        return transformed_documents

    def __call__(self, documents: List[Document]):
        batch_size = self.batch_size
        for i in range(0, len(documents), batch_size):

            List = documents[i : i + batch_size]
            print(i, len(List))
            self.process_batch(List)


# if __name__ == "__main__":
#     from lightrag.core.types import Document
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
#     ep = EmbeddingPipeline()
#     ep.setup_database_manager(postgres_url)

#     new_documents = [Document(**doc) for doc in documents]

#     ep(new_documents)
