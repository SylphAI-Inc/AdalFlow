from lightrag.database.sqlalchemy.sqlachemy_manager import DatabaseManager

# we need to import the DocumentModel class to create the tables
from lightrag.database.sqlalchemy.model import DocumentModel  # noqa


if __name__ == "__main__":
    db_name = "vector_db"
    postgres_url = f"postgresql://postgres:password@localhost:5432/{db_name}"

    db_manager = DatabaseManager(postgres_url)
    db_manager.create_tables()
    print(db_manager.list_table_schemas())
    db_manager.close()
