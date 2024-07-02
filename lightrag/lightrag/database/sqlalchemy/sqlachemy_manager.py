from typing import Any, Dict

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import text
from sqlalchemy.engine import reflection

from lightrag.database.sqlalchemy.base import Base
from lightrag.database.sqlalchemy.model import *  # noqa


class DatabaseManager:
    def __init__(self, database_url: str):
        self._engine = create_engine(database_url)
        self._Session = sessionmaker(bind=self._engine)

    @property
    def engine(self):
        return self._engine

    def get_database_name(self):
        # This method will need to be adjusted depending on the database type
        if "postgresql" in self._engine.url.drivername:
            query = "SELECT current_database()"
        elif "mysql" in self._engine.url.drivername:
            query = "SELECT database()"
        elif "sqlite" in self._engine.url.drivername:
            return (
                self._engine.url.database
            )  # Directly return the database file for SQLite
        else:
            raise Exception("Unsupported database type")

        with self._engine.connect() as connection:
            result = connection.execute(text(query))
            return result.scalar()

    def create_tables(self):
        r"""Creates the tables in the database."""
        Base.metadata.create_all(self.engine)

    def list_database_schemas(self) -> Dict[str, Any]:
        r"""List the database schemas."""
        inspector = inspect(self.engine)
        return {"schemas": inspector.get_schema_names()}

    def list_table_schemas(self) -> Dict[str, Any]:
        r"""List the table schemas from the database (engine)."""
        inspector = reflection.Inspector.from_engine(self.engine)

        # Retrieve all table names
        table_names = inspector.get_table_names()

        # Initialize a dictionary to hold table details
        table_details = {}

        # Loop through each table name to fetch schema details
        for table_name in table_names:
            # Get columns for each table
            columns = inspector.get_columns(table_name)
            table_details[table_name] = {"columns": columns}

        # Optionally, you can also fetch and add other details like foreign keys, indexes, etc.
        for table_name in table_names:
            foreign_keys = inspector.get_foreign_keys(table_name)
            indexes = inspector.get_indexes(table_name)
            table_details[table_name]["foreign_keys"] = foreign_keys
            table_details[table_name]["indexes"] = indexes
        return table_details

    def get_session(self) -> Session:
        return self._Session()

    def execute_query(self, query: str, params: dict = None) -> None:
        r"""Executes a raw SQL query."""
        result_list = []
        with self.get_session() as session:
            result = session.execute(text(query), params)
            session.commit()
            # Use result.keys() to get the column names
            result_list = [
                {key: value for key, value in zip(result.keys(), row)} for row in result
            ]
        return result_list

    def search(self, model, **kwargs):
        with self.get_session() as session:
            result = session.query(model).filter_by(**kwargs).all()
        return result

    def close(self):
        self._engine.dispose()
        self._Session.close_all()
