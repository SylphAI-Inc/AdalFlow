r"""
DB component abstracts both local and external database for other components such as Retriever to retrieve data from.
These db can stream or read its documents and pass through a sequential data transformer to convert to the required format.

For cloud db, you can additionally add a manager to CRUD the data. These are data processing and you can still
use some functional components in lightrag to process the data.
"""

from typing import List, Optional
from core.component import Component, Sequential
from core.documents_data_class import Document, Chunk
from core.document_splitter import DocumentSplitter
from core.data_components import ToEmbeddings


class LocalDocumentDB(Component):
    r"""
    For now we use a list of Documents, might consider optimize it later
    """

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        data_transformer: Optional[Component] = None,
    ):
        super().__init__()
        self.documents = documents if documents else []
        # data_transformer = Sequential(DocumentSplitter())
        self.data_transformer = data_transformer
        self.transformed_documents = []

    def load_documents(self, documents: List[Document]):
        self.documents = documents

    def __call__(self) -> Optional[List[Document]]:
        if self.data_transformer:
            self.transformed_documents = self.data_transformer(self.documents)
            return self.transformed_documents
