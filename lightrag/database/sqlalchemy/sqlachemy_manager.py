from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import text

Base = declarative_base()


class DatabaseManager:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        return self.Session()

    def execute_query(self, query: str, params: dict = None) -> None:
        with self.get_session() as session:
            result = session.execute(text(query), params)
            session.commit()
        return result

    def search(self, model, **kwargs):
        with self.get_session() as session:
            result = session.query(model).filter_by(**kwargs).all()
        return result
