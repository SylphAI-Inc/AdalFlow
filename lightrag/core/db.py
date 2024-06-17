"""LocalDocumentDB is to handle documents: in-memory and stored in the local file system in pickle format."""

from typing import List, Optional, Callable, Dict, Any, Type
import pickle
import os
import logging
from dataclasses import field, dataclass
import threading


from lightrag.core.component import Component
from lightrag.core.types import Document
from lightrag.core.functional import generate_readable_key_for_function

"""
Why do we need a localDocumentDB as the product db is always in the cloud?

1. For testing and development, we can use a local db to test the components and experimenting before deploying to the cloud.

This means localdb has to be highly flexible and customizable and will eventullay in sync with the cloud db.

So a great local db is highly important and the #1 step to build a product.

A dataset can include anything, and some parts will be represented as local document db.
"""


log = logging.getLogger(__name__)


# @dataclass
@dataclass
class LocalDocumentDB:
    r"""
    It inherits from the Component class for better structure visualization. But normally it cant be chained as part of the query flow/pipeline.
    For now we use a list of Documents, might consider optimize it later

    LocalDocumentDB is used to experiment and once deployed, can be used to manage per-query related documents in the application.
    It can be used to help retriever prepare index using transformer, and to store and load back the index to avoid re-computation.

    Retriever will be configured already, but when we retrieve, we can potentially override the initial configuration.
    """

    documents: List[Document] = field(
        default_factory=list, metadata={"description": "The original documents"}
    )

    transformed_documents: Dict[str, List[Document]] = field(
        default_factory=dict, metadata={"description": "Transformed documents by key"}
    )
    mapped_documents: Dict[str, List[Document]] = field(
        default_factory=dict, metadata={"description": "Mapped documents by key"}
    )
    transformer_setups: Dict[str, Any] = field(
        default_factory=dict, metadata={"description": "Transformer setup by key"}
    )

    # def __post_init__(self):
    #     super().__init__()

    def list_mapped_data_keys(self) -> List[str]:
        return list(self.mapped_documents.keys())

    def list_transformed_data_keys(self) -> List[str]:
        return list(self.transformed_documents.keys())

    def map_data(
        self,
        key: Optional[str] = None,
        map_func: Callable[[Document], Document] = lambda x: x,
    ) -> List[Document]:
        """Map a document especially the text field into the content that you want other components to use such as document splitter and embedder."""
        # make a copy of the documents
        if key is None:
            key = generate_readable_key_for_function(map_func)
        documents_to_use = self.documents.copy()
        self.mapped_documents[key] = [
            map_func(document) for document in documents_to_use
        ]
        return key

    def get_mapped_data(self, key: str) -> List[Document]:
        return self.mapped_documents[key]

    def _get_transformer_name(self, transformer: Component) -> str:
        # check all _sub_components and join the name with _
        name = f"{transformer.__class__.__name__}_"
        for n, _ in transformer.named_components():
            name += n + "_"
        return name

    def register_transformer(self, transformer: Component, key: Optional[str] = None):
        """Register a transformer to be used later for transforming the documents."""
        if key is None:
            key = self._get_transformer_name(transformer)
            log.info(f"Generated key for transformer: {key}")
        self.transformer_setups[key] = transformer

    def transform_data(
        self,
        transformer: Component,
        key: Optional[str] = None,
        # documents: Optional[List[Document]] = None,
    ) -> List[Document]:
        """Transform the documents using the transformer, the transformed documents will be used to build index."""
        if key is None:
            key = self._get_transformer_name(transformer)
            log.info(f"Generated key for transformed data: {key}")

        documents_to_use = self.documents.copy()
        self.transformed_documents[key] = transformer(documents_to_use)
        self.transformer_setups[key] = transformer  # Record the transformer obj
        return key

    def get_transformed_data(self, key: str) -> List[Document]:
        """Get the transformed documents by key."""
        return self.transformed_documents[key]

    # TODO: when new documents are added, we need to extend the transformed documents as well
    def load_documents(
        self, documents: List[Document], apply_transformer: bool = False
    ):
        """Load the db with new documents."""
        if apply_transformer:
            for key, transformer in self.transformer_setups.items():
                self.transformed_documents[key] = transformer(documents)
        self.documents = documents

    def extend_documents(
        self, documents: List[Document], apply_transformer: bool = False
    ):
        """Extend the db with new documents."""
        if apply_transformer:
            for key, transformer in self.transformer_setups.items():
                self.transformed_documents[key].extend(transformer(documents))

        self.documents.extend(documents)

    def reset_all(self):
        r"""Reset all attributes to empty."""
        self.reset_documents()
        self.mapped_documents = {}
        self.transformer_setups = {}

    def reset_documents(self, remove_transformed=True):
        """Remove all documents.
        Args:
            remove_transformed (bool): Whether to remove the transformed documents as well.
        """
        self.documents = []
        if remove_transformed:
            self.transformed_documents = {}

    def remove_documents_by_index(self, indexes: List[int], remove_transformed=True):
        """Remove documents by index.
        Args:
            indexes (List[int]): List of indexes to remove.
            remove_transformed (bool): Whether to remove the transformed documents as well.
        """
        if remove_transformed:
            for key in self.transformed_documents.keys():
                self.transformed_documents[key] = [
                    doc
                    for i, doc in enumerate(self.transformed_documents[key])
                    if i not in indexes
                ]
        for index in indexes:
            self.documents.pop(index)

    def save_state(self, filepath: str):
        """Save the current state (attributes) of the document DB using pickle.

        Note:
            The transformer setups will be lost when pickling. As it might not be picklable.
        """
        filepath = filepath or "storage/local_document_db.pkl"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_state(cls, filepath: str = None) -> "LocalDocumentDB":
        """Load the state of the document DB from a pickle file."""
        filepath = filepath or "storage/local_document_db.pkl"
        with open(filepath, "rb") as file:
            return pickle.load(file)

    # transformer set up will be lost when pickling
    def __getstate__(self):
        """Exclude non-picklable attributes."""

        state = self.__dict__.copy()
        # Remove the transformer setups
        state.pop("transformer_setups", None)
        return state

    def __setstate__(self, state):
        """Restore state (including non-picklable attributes with default values if needed)."""
        self.__dict__.update(state)
        # Reinitialize transformer setups to an empty dictionary
        self.transformer_setups = {}


# if __name__ == "__main__":
#     from lightrag.core.types import Document
#     from lightrag.components.retriever import FAISSRetriever
#     from lightrag.core.db import LocalDocumentDB
#     from lightrag.utils.config import construct_components_from_config
#     from lightrag.utils import setup_env

#     data_transformer_config = {  # attribute and its config to recreate the component
#         "embedder": {
#             "component_name": "Embedder",
#             "component_config": {
#                 "model_client": {
#                     "component_name": "OpenAIClient",
#                     "component_config": {},
#                 },
#                 "model_kwargs": {
#                     "model": "text-embedding-3-small",
#                     "dimensions": 256,
#                     "encoding_format": "float",
#                 },
#             },
#         },
#         "document_splitter": {
#             "component_name": "DocumentSplitter",
#             "component_config": {
#                 "split_by": "word",
#                 "split_length": 400,
#                 "split_overlap": 200,
#             },
#         },
#         "to_embeddings": {
#             "component_name": "ToEmbeddings",
#             "component_config": {
#                 "vectorizer": {
#                     "component_name": "Embedder",
#                     "component_config": {
#                         "model_client": {
#                             "component_name": "OpenAIClient",
#                             "component_config": {},
#                         },
#                         "model_kwargs": {
#                             "model": "text-embedding-3-small",
#                             "dimensions": 256,
#                             "encoding_format": "float",
#                         },
#                     },
#                     # the other config is to instantiate the entity (class and function) with the given config as arguments
#                     # "entity_state": "storage/embedder.pkl", # this will load back the state of the entity
#                 },
#                 "batch_size": 100,
#             },
#         },
#     }

#     path = "developer_notes/developer_notes/db_states.pkl"
#     db = LocalDocumentDB.load_state(path)
#     transformer_key = db.list_transformed_data_keys()[0]
#     print(db.transformer_setups)
#     components = construct_components_from_config(data_transformer_config)
#     embedder = components["embedder"]
#     transformed_documents = db.get_transformed_data(
#         transformer_key
#     )  # list of documents
#     embeddings = [doc.vector for doc in transformed_documents]
#     retriever = FAISSRetriever(
#         embedder=embedder, documents=embeddings
#     )  # allow to initialize with documents too

#     query = "What happened at Viaweb and Interleaf?"
#     retrieved_documents = retriever(query)
#     print(retrieved_documents)
