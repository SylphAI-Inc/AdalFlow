r"""
DB component abstracts both local and external database for other components such as Retriever to retrieve data from.
These db can stream or read its documents and pass through a sequential data transformer to convert to the required format.

For cloud db, you can additionally add a manager to CRUD the data. These are data processing and you can still
use some functional components in lightrag to process the data.
"""

from typing import List, Optional, Union, Any
from core.component import Component
from core.data_classes import Document

from core.retriever import Retriever, RetrieverInputType, RetrieverOutputType

"""
Why do we need a localDocumentDB as the product db is always in the cloud?

1. For testing and development, we can use a local db to test the components and experimenting before deploying to the cloud.

This means localdb has to be highly flexible and customizable and will eventullay in sync with the cloud db.

So a great local db is highly important and the #1 step to build a product.

A dataset can include anything, and some parts will be represented as local document db.
"""


class LocalDocumentDB(Component):
    r"""
    It inherits from the Component class for better structure visualization. But normally it cant be chained as part of the query flow/pipeline.
    For now we use a list of Documents, might consider optimize it later

    Retriever will be configured already, but when we retrieve, we can potentially override the initial configuration.
    """

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        data_transformer: Optional[Component] = None,
        retriever: Optional[
            Retriever
        ] = None,  # retriever can be stateless and should have build_index method, attach a retriever to the db
        retriever_output_processors: Optional[
            Component
        ] = None,  # any output processors to be applied to the retrieved documents, such as deduplication if query expansion is used
    ):
        super().__init__()
        self.documents = documents if documents else []  # the original documents
        # data_transformer = Sequential(DocumentSplitter())
        # TODO: how is data_transformer fit into the pipeline?, should it do in-place transformation?
        self.data_transformer = data_transformer
        self.transformed_documents = []
        self.retriever = retriever
        self.retrieve_output_processors = retriever_output_processors

    def attach_retriever(self, retriever: Component) -> None:
        self.retriever = retriever

    # TODO: load documents from local folders
    # TODO: delete documents or modify documents
    # TODO: persist the documents to the local folders
    # langchain you have to initi documents at first, which does not make sense in a pipeline
    # we use load_documetns, extend, reset or even delete some documents
    def load_documents(self, documents: List[Document]):
        self.documents = documents

    def extend_documents(self, documents: List[Document]):
        self.documents.extend(documents)

    def reset_documents(self):
        self.documents = []

    def build_retrieve_index(self):
        documents_to_use = self.documents
        if self.transformed_documents:
            documents_to_use = self.transformed_documents
        self.retriever.build_index_from_documents(documents_to_use)

    # TODO: allow better otuput type specification and type hinting
    def retrieve(self, query_or_queries: RetrieverInputType) -> Any:
        # check if the retriever has build index or not
        if not self.retriever.indexed:
            raise ValueError("Retriever is not indexed, please call build_index first")
        response: RetrieverOutputType = self.retriever(query_or_queries)
        # set the actual documents to the response
        for i, output in enumerate(response):
            # convert doc_indexes to doc_contents
            response[i].chunks = [
                self.documents[doc_index] for doc_index in output.doc_indexes
            ]
        if self.retrieve_output_processors:
            response = self.retrieve_output_processors(response)
        return response

    def __call__(self) -> Optional[List[Document]]:
        if self.data_transformer:
            self.transformed_documents = self.data_transformer(self.documents)
            return self.transformed_documents
