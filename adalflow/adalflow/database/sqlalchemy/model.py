"""Define the data model for the SQLAlchemy ORM. This is the schema for the database.

We use sqlalchemy as it is a popular ORM for Python and supports a wide range of databases.

For example:

# Connect to PostgreSQL
engine = create_engine('postgresql://user:password@localhost/mydatabase')

# Connect to MySQL
engine = create_engine('mysql+pymysql://user:password@localhost/mydatabase')

# Connect to SQLite (local file)
engine = create_engine('sqlite:///path/to/database.db')

# Connect to SQLite (in-memory)
engine = create_engine('sqlite:///:memory:')

We define the schema for the database using the declarative_base class provided by sqlalchemy. This allows us to define our data model as a class and then create the database schema from it.

Note:
    - This script might only be applicable to PostgreSQL databases along with the pgvector extension.
    - Ensure you have set up the right database and that it supports the pgvector extension, either local or cloud.

References:
[1] https://github.com/pgvector/pgvector?tab=readme-ov-file#installation-notes---linux-and-mac
[2] pgvector python package: https://pypi.org/project/pgvector/
"""

from typing import Optional, Dict, List
from datetime import datetime
import logging
from adalflow.utils.lazy_import import safe_import, OptionalPackages

sqlalchemy = safe_import(
    OptionalPackages.SQLALCHEMY.value[0], OptionalPackages.SQLALCHEMY.value[1]
)
pgvector = safe_import(
    OptionalPackages.PGVECTOR.value[0], OptionalPackages.PGVECTOR.value[1]
)
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    TIMESTAMP,
    func,
)
import uuid


from sqlalchemy.orm import Session as BaseSession
from sqlalchemy.dialects.postgresql import JSONB

from pgvector.sqlalchemy import Vector

from adalflow.database.sqlalchemy.base import Base


log = logging.getLogger(__name__)


class DocumentBase(Base):
    # __tablename__ = "document"
    __table_args__ = {"extend_existing": True}  # Allows extending the table
    __abstract__ = True
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    text = Column(Text, nullable=False)

    meta_data = Column(
        JSONB
    )  # Utilizing JSONB for metadata storage if using PostgreSQL
    order = Column(Integer)
    parent_doc_id = Column(String)
    estimated_num_tokens = Column(Integer)
    created_at = Column(
        DateTime, default=datetime.utcnow
    )  # Sets the timestamp on record creation
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )  # Updates timestamp on any update

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "meta_data": self.meta_data,
            "order": self.order,
            "parent_doc_id": self.parent_doc_id,
            "estimated_num_tokens": self.estimated_num_tokens,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def insert_update_single(cls, document_data: Dict, session: BaseSession):
        r"""Insert or update a single document in the database."""
        identifier = document_data.get("id", None)
        existing_document = None
        if identifier:  # we can check existing documents
            existing_document = session.query(cls).filter_by(id=identifier).first()

        try:
            if existing_document:
                for key, value in document_data.items():
                    setattr(existing_document, key, value)
                log.debug(f"Updated document: {existing_document}")
            else:
                new_document = cls(**document_data)
                session.add(new_document)
                log.debug(f"Inserted document: {new_document}")
            session.commit()
            return existing_document if existing_document else new_document
        except Exception as e:
            log.error(f"Error: {e}, Rolling back transaction.")
            session.rollback()
            return None

    @classmethod
    def insert_update_bulk(cls, document_datas: List[Dict], session: BaseSession):
        r"""Insert or update multiple documents in the database."""
        try:
            identifiers = [data["id"] for data in document_datas if "id" in data]
            existing_documents = {
                doc.id: doc
                for doc in session.query(cls).filter(cls.id.in_(identifiers)).all()
            }

            for document_data in document_datas:
                existing_document = existing_documents.get(document_data["id"])
                if existing_document:
                    for key, value in document_data.items():
                        setattr(existing_document, key, value)
                else:
                    new_document = cls(**document_data)
                    session.add(new_document)

            session.commit()
            print("Bulk insert/update successful.")
            log.debug("Bulk insert/update successful.")
        except Exception as e:
            print(f"Error during bulk insert/update: {e}")
            log.error(f"Error: {e}, Rolling back transaction for bulk insert/update.")
            session.rollback()
            return None


# TODO: extend more fields to support full-text search
# class DocumentSearchExtended(DocumentBase):
#     __abstract__ = True
#     text_tsvector = Column("text_tsvector", String)


class DocumentWithoutVector(DocumentBase):
    __doc__ = r"""Document class without pgvector support."""
    __tablename__ = "document"
    __bind_key__ = "document_db"
    __table_args__ = {"extend_existing": True}  # Allows extending the table

    def __init__(
        self,
        table_name: Optional[str] = None,
        bind_key: Optional[str] = None,
        *args,
        **kwargs,
    ):
        r"""Initialize the DocumentWithoutVector class.

        We allow you to specify the table name and bind key for the document class.
        """
        if table_name:
            type(self).__tablename__ = table_name
        if bind_key:
            type(self).__bind_key__ = bind_key
        super().__init__(*args, **kwargs)


class DocumentModel(DocumentBase):
    __doc__ = r"""Document class extending the DocumentBase with pgvector support."""

    __tablename__ = "document"
    __bind_key__ = "vector_db"

    __table_args__ = {"extend_existing": True}  # Allows extending the table

    vector = Column(Vector())  # using pgvector for vector storage

    def __init__(
        self,
        table_name: Optional[str] = None,
        bind_key: Optional[str] = None,
        *args,
        **kwargs,
    ):
        if table_name:
            type(self).__tablename__ = table_name
        if bind_key:
            type(self).__bind_key__ = bind_key
        super().__init__(*args, **kwargs)


class DialogueTurnModel(Base):
    __tablename__ = "dialogue_turn_all_in_one"
    __bind_key__ = "dialogue_db"

    id = Column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )  # Assuming SERIAL is auto-incrementing integer
    turn_index = Column(Integer)
    user_id = Column(
        String,
        nullable=False,
    )
    session_id = Column(
        String,
        # ForeignKey("dialogue_session.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_query_params = Column(JSONB)
    converse_agent_response = Column(JSONB)
    search_agent_response = Column(JSONB)
    user_feedback = Column(JSONB)
    turn_start_time = Column(DateTime, default=datetime.now())
    converse_agent_response_time = Column(DateTime, default=datetime.now())
    search_agent_response_time = Column(DateTime, default=datetime.now())  # matches
    # matches should be a list of dictionaries
    created_at = Column(TIMESTAMP, default=func.now())
    updated_at = Column(TIMESTAMP, default=func.now(), onupdate=func.now())
    deleted_at = Column(TIMESTAMP, nullable=True)  # soft delete

    def __str__(self):
        str_repr = ""
        str_repr += f"id: {self.id}\n"
        str_repr += f"turn_index: {self.turn_index}\n"
        str_repr += f"user_id: {self.user_id}\n"
        str_repr += f"session_id: {self.session_id}\n"
        str_repr += f"user_query_params: {self.user_query_params}\n"
        str_repr += f"converse_agent_response: {self.converse_agent_response}\n"
        str_repr += f"search_agent_response: {self.search_agent_response}\n"
        str_repr += f"user_feedback: {self.user_feedback}\n"
        str_repr += f"turn_start_time: {self.turn_start_time}\n"
        str_repr += (
            f"converse_agent_response_time: {self.converse_agent_response_time}\n"
        )
        str_repr += f"search_agent_response_time: {self.search_agent_response_time}\n"
        return str_repr

    def to_dict(self):
        return {
            "turn_index": self.turn_index,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "user_query_params": self.user_query_params,
            "converse_agent_response": self.converse_agent_response,
            "search_agent_response": self.search_agent_response,
            "user_feedback": self.user_feedback,
            "turn_start_time": self.turn_start_time,
            "converse_agent_response_time": self.converse_agent_response_time,
            "search_agent_response_time": self.search_agent_response_time,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted_at": self.deleted_at,
        }


# if __name__ == "__main__":
#     from sqlalchemy.sql import text

#     from adalflow.database.sqlalchemy.sqlachemy_manager import DatabaseManager

#     db_name = "vector_db"
#     postgres_url = f"postgresql://postgres:password@localhost:5432/{db_name}"

#     db_manager = DatabaseManager(postgres_url)
#     engine = db_manager.engine
#     table_names = db_manager.list_table_schemas()
#     print(f"Table names: {table_names}")
#     database_schema = db_manager.list_database_schemas()
#     print(f"Database schema: {database_schema}")
#     database_name = db_manager.get_database_name()
#     print(f"Database name: {database_name}")
#     # engine = create_engine(postgres_url)

#     try:
#         # Begin a transaction
#         with engine.connect() as connection:
#             # Use the connection to execute a raw SQL command
#             # Ensuring the session commits after, especially for operations requiring administrative rights
#             connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
#             connection.commit()  # Explicitly commit to ensure that the CREATE EXTENSION command is finalized
#     except Exception as e:
#         print(f"An error occurred while trying to create the extension: {e}")

#     # Base.metadata.create_all(engine, checkfirst=True)
#     # Session = sessionmaker(bind=engine)
#     # session = Session()
#     db_manager.create_tables()
#     session = db_manager.get_session()

#     # doc = Document(text="Hello world")
#     DocumentModel.insert_update_single({"text": "Hello world"}, session=session)
#     # do bulk insert
#     DocumentModel.insert_update_bulk(
#         [
#             {"id": "doc2", "text": "Hello world 2"},
#             {"id": "doc3", "text": "Hello world 3"},
#         ],
#         session=session,
#     )
#     db_manager.close()
#     # session.add(doc)
#     # session.commit()
#     # from adalflow.core.types import Document as LightragDocument

#     # doc_2 = LightragDocument.from_dict(doc.to_dict())
#     # print("doc:", doc, doc.id)
#     # print("doc_2:", doc_2, doc_2.id)

#     # print(doc.id)
#     # print(doc.text)
#     # session.close()
#     # engine.dispose()
