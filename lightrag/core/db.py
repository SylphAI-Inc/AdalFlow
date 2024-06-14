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


class TransformerRegistry:
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, transformer_cls: Type):
        cls._registry[name] = transformer_cls

    @classmethod
    def get(cls, name: str) -> Type:
        return cls._registry[name]

    @classmethod
    def get_all(cls) -> Dict[str, Type]:
        return cls._registry


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

    def register_transformer(self, transformer: Component, key: Optional[str] = None):
        """Register a transformer to be used later for transforming the documents."""
        if key is None:
            key = transformer._get_name() + "_transformer"
            log.info(f"Generated key for transformer: {key}")
        self.transformer_setups[key] = transformer

    def transform_data(
        self,
        transformer: Component,
        key: Optional[str] = None,
        documents: Optional[List[Document]] = None,
    ) -> List[Document]:
        """Transform the documents using the transformer, the transformed documents will be used to build index."""
        if key is None:
            key = transformer._get_name() + "_transformer"
            log.info(f"Generated key for transformed data: {key}")

        documents_to_use = documents.copy() if documents else self.documents.copy()
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
        """Save the current state (attributes) of the document DB using pickle."""
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

    documents = [
        Document(text="This is a test document. It is a long document."),
        Document(text="This is another test document. It is also a long document."),
    ]
    db = LocalDocumentDB()
    print(db)
    db.load_documents(documents)
    print(db)

    from lightrag.components.model_client import OpenAIClient
    from lightrag.core.document_splitter import DocumentSplitter
    from lightrag.core.embedder import Embedder, BatchEmbedder
    from lightrag.core.component import Sequential
    from lightrag.core.data_components import ToEmbeddings
    from lightrag.utils import setup_env
    from lightrag.utils.registry import EntityMapping

    # # Register transformers
    # TransformerRegistry.register("DocumentSplitter", DocumentSplitter)
    # TransformerRegistry.register("Embedder", Embedder)

    # this relatest to the config
    # TODO: component needs to be picklable
    config = {  # attribute and its config to recreate the component
        "embedder": {
            "entity_name": "Embedder",
            "entity_config": {
                "model_client": {
                    "entity_name": "OpenAIClient",
                },
                "model_kwargs": {
                    "model": "text-embedding-3-small",
                    "dimensions": 256,
                    "encoding_format": "float",
                },
            },
        },
        "document_splitter": {
            "entity_name": "DocumentSplitter",
            "entity_config": {
                "split_by": "word",
                "split_length": 400,
                "split_overlap": 200,
            },
        },
        "to_embeddings": {
            "entity_name": "ToEmbeddings",
            "entity_config": {
                "vectorizer": "embedder",
                "batch_size": 100,
            },
        },
    }

    # TODO: create config from a component pipeline.

    def construct_entity(config: Dict[str, Any]) -> Any:
        entity_name = config["entity_name"]
        entity_cls = EntityMapping.get(entity_name)
        entity_config = config.get("entity_config", {})

        initialized_config = {}

        for key, value in entity_config.items():
            if isinstance(value, dict) and "entity_name" in value:
                # Recursively construct sub-entities
                initialized_config[key] = construct_entity(value)
            else:
                initialized_config[key] = value

        return entity_cls(**initialized_config)

    def construct_entities_from_config(
        config: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        entities = {}
        for attr, entity_config in config.items():
            entities[attr] = construct_entity(entity_config)
        return entities

    vectorizer_settings = {
        "batch_size": 100,
        "model_kwargs": {
            "model": "text-embedding-3-small",
            "dimensions": 256,
            "encoding_format": "float",
        },
    }
    retriever_settings = {
        "top_k": 2,
    }
    generator_model_kwargs = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.3,
        "stream": False,
    }
    text_splitter_settings = {  # TODO: change it to direct to spliter kwargs
        "split_by": "word",
        "chunk_size": 400,
        "chunk_overlap": 200,
    }
    vectorizer = Embedder(
        model_client=OpenAIClient(),
        # batch_size=self.vectorizer_settings["batch_size"],
        model_kwargs=vectorizer_settings["model_kwargs"],
    )
    # TODO: check document splitter, how to process the parent and order of the chunks
    text_splitter = DocumentSplitter(
        split_by=text_splitter_settings["split_by"],
        split_length=text_splitter_settings["chunk_size"],
        split_overlap=text_splitter_settings["chunk_overlap"],
    )
    batch_embedder = BatchEmbedder(
        embedder=vectorizer,
        batch_size=vectorizer_settings["batch_size"],
    )
    data_transformer = Sequential(
        text_splitter,
        ToEmbeddings(
            vectorizer=vectorizer,
            batch_size=vectorizer_settings["batch_size"],
        ),
    )
    entities = construct_entities_from_config(config)
    print("entities:", entities)

    # db = LocalDocumentDB.load_state("storage/local_document_db.pkl")
    # print(db)
    db.register_transformer(data_transformer)
    # print(db.transformer_setups)
    print(db)
    db.transform_data(data_transformer)
    print(db)
    db.save_state("storage/local_document_db.pkl")
