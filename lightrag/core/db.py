"""LocalDB to perform in-memory storage and data persistence(pickle or any filesystem) for data models like documents and dialogturn."""

from typing import List, Optional, Callable, Dict, Any, TypeVar, Generic
import pickle
import os
import logging
from dataclasses import field, dataclass


from lightrag.core.component import Component
from lightrag.core.types import Document
from lightrag.core.functional import generate_readable_key_for_function


log = logging.getLogger(__name__)

T = TypeVar("T")

U = TypeVar("U")  # U will be the type after transformation


# @dataclass
# TODO: make it work with any data type just like a cloud db that can have any table with different columns
@dataclass
class LocalDB(Generic[T]):
    __doc__ = r"""LocalDB with in-memory CRUD operations, data persistence, data mapping and transformation.

    Args:

        items (List[T], optional): The original data items. Defaults to [].
        transformed_items (Dict[str, List [U]], optional): Transformed data items by key. Defaults to {}.
             Transformer must be of type Component.
        mapped_items (Dict[str, List[Document]], optional): Mapped documents by key. Defaults to {}.
             map_func is a function that maps the document to the desired format.

        transformer_setups (Dict[str, Component], optional): Transformer setup by key. Defaults to {}.
          It is used to save the transformer setup for later use.
    """
    name: Optional[str] = None
    items: List[T] = field(
        default_factory=list, metadata={"description": "The original documents"}
    )

    transformed_items: Dict[str, List[U]] = field(
        default_factory=dict, metadata={"description": "Transformed documents by key"}
    )
    mapped_items: Dict[str, List[Document]] = field(
        default_factory=dict, metadata={"description": "Mapped documents by key"}
    )

    transformer_setups: Dict[str, Component] = field(
        default_factory=dict, metadata={"description": "Transformer setup by key"}
    )
    mapper_setups: Dict[str, Callable[[Document], Document]] = field(
        default_factory=dict, metadata={"description": "Mapper setup by key"}
    )

    # def __post_init__(self):
    #     super().__init__()
    @property
    def length(self):
        return len(self.items)

    def list_mapped_data_keys(self) -> List[str]:
        return list(self.mapped_items.keys())

    def list_transformed_data_keys(self) -> List[str]:
        return list(self.transformed_items.keys())

    def map_data(
        self,
        key: Optional[str] = None,
        map_fn: Callable[[Document], Document] = lambda x: x,
    ) -> List[Document]:
        """Map a document especially the text field into the content that you want other components to use such as document splitter and embedder."""
        # make a copy of the documents
        if key is None:
            key = generate_readable_key_for_function(map_fn)
        documents_to_use = self.items.copy()
        self.mapped_items[key] = [map_fn(document) for document in documents_to_use]
        self.mapper_setups[key] = map_fn
        return key

    def get_mapped_data(self, key: str) -> List[Document]:
        return self.mapped_items[key]

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
        map_fn: Optional[
            Callable[[T], Any]
        ] = None,  # to map the document to designed format the transformer expects
        # documents: Optional[List[Document]] = None,
    ) -> List[U]:
        """Transform the documents using the transformer, the transformed documents will be used to build index."""
        if key is None:
            key = self._get_transformer_name(transformer)
            log.info(f"Generated key for transformed data: {key}")

        if map_fn is not None:
            items_to_use = [map_fn(item) for item in self.items]
        else:
            items_to_use = self.items.copy()

        self.transformed_items[key] = transformer(items_to_use)
        self.transformer_setups[key] = transformer  # Record the transformer obj
        return key

    def get_transformed_data(self, key: str) -> List[Document]:
        """Get the transformed documents by key."""
        return self.transformed_items[key]

    # TODO: when new documents are added, we need to extend the transformed documents as well
    def load(self, documents: List[Any], apply_transformer: bool = True):
        """Load the db with new documents."""
        if apply_transformer:
            for key, transformer in self.transformer_setups.items():
                self.transformed_items[key] = transformer(documents)
        self.items = documents

    def extend(self, documents: List[Any], apply_transformer: bool = True):
        """Extend the db with new documents."""
        if apply_transformer:
            for key, transformer in self.transformer_setups.items():
                self.transformed_items[key].extend(transformer(documents))

        self.items.extend(documents)

    def delete(self, indices: List[int], remove_transformed=True):
        """Remove documents by index.
        Args:
            indices (List[int]): List of indices to remove.
            remove_transformed (bool): Whether to remove the transformed documents as well.
        """
        if remove_transformed:
            for key in self.transformed_items.keys():
                self.transformed_items[key] = [
                    doc
                    for i, doc in enumerate(self.transformed_items[key])
                    if i not in indices
                ]
        for index in indices:
            self.items.pop(index)

    def add(
        self, document: Any, index: Optional[int] = None, apply_transformer: bool = True
    ):
        """Add a single document to the db.

        Args:
            document (Any): The document to add.
            index (int, optional): The index to add the document at. Defaults to None.
            When None, the document is appended to the end.
            apply_transformer (bool, optional): Whether to apply the transformer to the document. Defaults to True.
        """
        transformed_documents = {}
        if apply_transformer:
            for key, transformer in self.transformer_setups.items():
                transformed_documents[key] = transformer(document)
        # add the document
        if index is not None:
            self.items.insert(index, document)
            for key, transformed_docs in transformed_documents.items():
                self.transformed_items[key].insert(index, transformed_docs)
        else:
            self.items.append(document)
            for key, transformed_docs in transformed_documents.items():
                self.transformed_items[key].append(transformed_docs)

    def reset_all(self):
        r"""Reset all attributes to empty."""
        self.reset()
        self.mapped_items = {}
        self.transformer_setups = {}

    def reset(self, remove_transformed=True):
        """Remove all documents.
        Args:
            remove_transformed (bool): Whether to remove the transformed documents as well.
        """
        self.items = []
        if remove_transformed:
            self.transformed_items = {}

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
    def load_state(cls, filepath: str = None) -> "LocalDB":
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


if __name__ == "__main__":
    from lightrag.core.types import Document
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
