"""LocalDB to perform in-memory storage and data persistence(pickle or any filesystem) for data models like documents and dialogturn."""

from typing import List, Optional, Callable, Dict, Any, TypeVar, Generic
import logging
import os
from dataclasses import field, dataclass
import pickle


from lightrag.core.component import Component, Sequential
from lightrag.utils.registry import EntityMapping


log = logging.getLogger(__name__)

T = TypeVar("T")  # Allow any type as items

U = TypeVar("U")  # U will be the type after transformation


@dataclass
class LocalDB(Generic[T]):
    __doc__ = r"""LocalDB with in-memory CRUD operations, data transformation/processing pipelines, and persistence.

    LocalDB is highly flexible.
    1. It can store any type of data items in the `items` attribute.
    2. You can register and apply multiple transformers, and save the transformed data in the `transformed_items` attribute.
       This is highly useful to manage experiments with different data transformations.
    3. You can save the state of the LocalDB to a pickle file and load it back later. All states are restored.
        str(local_db.__dict__) == str(local_db_loaded.__dict__) should be True.

    .. note::
        The transformer should be of type Component. We made the effort in the library to make every component picklable.

    CRUD operations:
    1. Create a new db: ``db = LocalDB(name="my_db")``
    2. load: Load the db with data. ``db.load([{"text": "hello world"}, {"text": "hello world2"}])``
    3. extend: Extend the db with data. ``db.extend([{"text": "hello world3"}])``.
       In default, the transformer is applied and the transformed data is extended.
    4. add: Add a single document to the db. ``db.add({"text": "hello world4"})``.
       In default, the transformer is applied and the transformed data is added.
       Unless the transformed data keeps the same length as the original data, the insert operation does not mean insert after the last item.
    5. delete: Remove documents by index. ``db.delete([0])``.
    6. reset: Remove all documents. ``db.reset()``, including transformed_items and transformer_setups,and map_fn_setups.

    Data transformation:
    1. Register a transformer first and apply it later

    .. code-block:: python

            db.register_transformer(transformer, key="test", map_fn=map_fn)
            # load data
            db.load([{"text": "hello world"}, {"text": "hello world2"}], apply_transformer=True)

            # or load data first and apply transformer by key
            db.load([{"text": "hello world"}, {"text": "hello world2"}], apply_transformer=False)
            db.apply_transformer("test")

    2. Add a version of transformed data to the db along with the transformer.

    .. code-block:: python

            db.transform_data(transformer, key="test", map_fn=map_fn)

    Data persistence:
    1. Save the state of the db to a pickle file.

    .. code-block:: python

            db.save_state("storage/local_document_db.pkl")

    2. Load the state of the db from a pickle file.

    .. code-block:: python

            db2 = LocalDB.load_state("storage/local_document_db.pkl")

    3. Check if the loaded and original db are the same.

    .. code-block:: python

                str(db.__dict__) == str(db2.__dict__) # expect True

    Args:

        items (List[T], optional): The original data items. Defaults to []. Can be any type such as Document, DialogTurn, dict, text, etc.
            The only requirement is that they should be picklable/serializable.
        transformed_items (Dict[str, List [U]], optional): Transformed data items by key. Defaults to {}.
             Transformer must be of type Component.
        transformer_setups (Dict[str, Component], optional): Transformer setup by key. Defaults to {}.
          It is used to save the transformer setup for later use.
        map_fn_setups (Dict[str, Callable[[T], Any]], optional): Map function setup by key. Defaults to {}.
    """

    name: Optional[str] = field(
        default=None, metadata={"description": "Name of the DB"}
    )
    items: List[T] = field(
        default_factory=list, metadata={"description": "The original data items"}
    )

    transformed_items: Dict[str, List[U]] = field(
        default_factory=dict, metadata={"description": "Transformed data items by key"}
    )

    transformer_setups: Dict[str, Component] = field(
        default_factory=dict, metadata={"description": "Transformer setup by key"}
    )
    map_fn_setups: Dict[str, Callable[[T], Any]] = field(
        default_factory=dict, metadata={"description": "Map function setup by key"}
    )

    @property
    def length(self):
        return len(self.items)

    def get_transformer_keys(self) -> List[str]:
        return list(self.transformed_items.keys())

    def get_transformed_data(self, key: str) -> List[U]:
        """Get the transformed documents by key."""
        return self.transformed_items[key]

    def _get_transformer_name(self, transformer: Component) -> str:
        name = f"{transformer.__class__.__name__}_"
        for n, _ in transformer.named_components():
            name += n + "_"
        return name

    def register_transformer(
        self,
        transformer: Component,
        key: Optional[str] = None,
        map_fn: Optional[Callable[[T], Any]] = None,
    ) -> str:
        """Register a transformer to be used later for transforming the data."""
        if key is None:
            key = self._get_transformer_name(transformer)
            log.info(f"Generated key for transformer: {key}")
        self.transformer_setups[key] = transformer
        if map_fn is not None:
            self.map_fn_setups[key] = map_fn
        return key

    def apply_transformer(self, key: str):
        """Apply the transformer to the data."""
        map_fn = self.map_fn_setups.get(key, None)
        if map_fn is not None:
            items_to_use = [map_fn(item) for item in self.items]
        else:
            items_to_use = self.items.copy()
        self.transformed_items[key] = self.transformer_setups[key](items_to_use)

    def transform_data(
        self,
        transformer: Component,
        key: Optional[str] = None,
        map_fn: Optional[Callable[[T], Any]] = None,
    ) -> List[U]:
        """Transform the documents using the given transformer and register the transformer."""

        key = self.register_transformer(transformer, key, map_fn)

        if map_fn is not None:
            items_to_use = [map_fn(item) for item in self.items]
        else:
            items_to_use = self.items.copy()

        self.transformed_items[key] = transformer(items_to_use)
        return key

    def load(self, documents: List[Any], apply_transformer: bool = True):
        """Load the db with new documents and apply the registered transformer.

        Args:
            documents (List[Any]): The documents to load.
            apply_transformer (bool, optional): Whether to apply the transformer to the documents. Defaults to True.

        Examples:

        .. code-block:: python

            db = LocalDB()
            db.load([{"text": "hello world"}, {"text": "hello world2"}])
        """
        self.items = documents
        if apply_transformer:
            for key, _ in self.transformer_setups.items():
                self.apply_transformer(key)

    def extend(self, documents: List[Any], apply_transformer: bool = True):
        """Extend the db with new documents."""
        self.items.extend(documents)
        if apply_transformer:
            for key, _ in self.transformer_setups.items():
                # check if there was a map function registered
                transformed_documents = []
                if key in self.map_fn_setups:
                    map_fn = self.map_fn_setups[key]
                    transformed_documents = transformer(
                        [map_fn(doc) for doc in documents]
                    )
                else:
                    transformed_documents = transformer(documents)
                self.transformed_items[key].extend(transformed_documents)

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
        """Add a single document to the db, optionally at a specific index.

        .. note::
            The document will be transformed using the registered transformer.
            Only if the transformed data keeps the same length as the original data, the ``insert`` operation will work correctly.

        Args:
            document (Any): The document to add.
            index (int, optional): The index to add the document at. Defaults to None.
            When None, the document is appended to the end.
            apply_transformer (bool, optional): Whether to apply the transformer to the document. Defaults to True.
        """
        transformed_documents: Dict[str, List] = {}
        if apply_transformer:
            for key, transformer in self.transformer_setups.items():
                transformed_docs = []
                map_fn = self.map_fn_setups.get(key, None)
                if map_fn is not None:
                    transformed_docs = transformer([map_fn(document)])
                else:
                    transformed_docs = transformer([document])
                transformed_documents[key] = transformed_docs

        if index is not None:
            self.items.insert(index, document)
            for key, transformed_docs in transformed_documents.items():
                for doc in transformed_docs:
                    self.transformed_items[key].insert(index, doc)
        else:
            self.items.append(document)
            for key, transformed_docs in transformed_documents.items():
                self.transformed_items[key].extend(transformed_docs)

    def reset(self):
        r"""Reset all attributes to empty."""
        self.reset()
        self.mapped_items = {}
        self.transformer_setups = {}
        self.map_fn_setups = {}

    def save_state(self, filepath: str):
        """Save the current state (attributes) of the document DB using pickle.

        Note:
            The transformer setups will be lost when pickling. As it might not be picklable.
        """
        filepath = filepath or "storage/local_document_db.pkl"
        file_dir = os.path.dirname(filepath)
        if file_dir and file_dir != "":
            os.makedirs(file_dir, exist_ok=True)

        with open(filepath, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_state(cls, filepath: str = None) -> "LocalDB":
        """Load the state of the document DB from a pickle file."""
        filepath = filepath or "storage/local_document_db.pkl"
        with open(filepath, "rb") as file:
            return pickle.load(file)

    def __getstate__(self):
        """Exclude non-picklable attributes and prepare transformer setups for serialization."""
        state = self.__dict__.copy()
        _transformer_files = {}
        _transformer_type_names = {}

        for key, transformer in self.transformer_setups.items():
            transformer_file = f"{key}_transformer.pkl"
            transformer.pickle_to_file(transformer_file)
            _transformer_files[key] = transformer_file
            _transformer_type_names[key] = transformer.__class__.__name__
        state["transformer_setups"] = {}
        state["_transformer_files"] = _transformer_files
        state["_transformer_type_names"] = _transformer_type_names
        return state

    def __setstate__(self, state):
        """Restore state and load transformer setups from their respective files."""
        _transformer_files = state.pop("_transformer_files")
        _transformer_type_names = state.pop("_transformer_type_names")
        self.__dict__.update(state)
        for key, transformer_file in _transformer_files.items():
            class_type = EntityMapping.get(_transformer_type_names[key])

            self.transformer_setups[key] = class_type.load_from_pickle(transformer_file)


if __name__ == "__main__":
    # test LocalDB
    from lightrag.core.component import (
        Sequential,
        Component,
        fun_to_component,
    )

    db = LocalDB()
    db.load([{"text": "hello world"}, {"text": "hello world2"}])

    @fun_to_component
    def add(docs: List):
        print(f"docs: {docs}")
        for doc in docs:
            doc["text"] += " add"
        return docs

    class Add(Component):
        def __init__(self):
            super().__init__()

        def call(self, docs: List):
            print(f"docs: {docs}")
            for doc in docs:
                doc["text"] += " add"
            return docs

    class Minus(Component):
        def __init__(self):
            super().__init__()

        def call(self, docs: List):
            print(f"docs minus: {docs}")
            for doc in docs:
                doc["text"] += " minus"
            return docs

    @fun_to_component
    def minus(docs: List):
        print(f"docs: {docs}")
        for doc in docs:
            doc["text"] += " minus"
        return docs

    # transformer = Sequential(FunComponent(add), FunComponent(minus))

    transformer = Sequential(add, minus)

    db.transform_data(key="test", transformer=transformer)
    print(db.transformed_items["test"])
    db.save_state("storage/local_document_db.pkl")
    db2 = LocalDB.load_state("storage/local_document_db.pkl")
    print(db2)
    # db.save_state("storage/local_document_db.pkl")
    # db2 = LocalDB.load_state("storage/local_document_db.pkl")
    # print(db2.transformed_items["test"])

    # # use the transformerp
    # transformer_2 = db2.transformer_setups["test"]

    # print(f"typeof transformer_2: {type(transformer_2)}")

    # print(transformer_2([{"text": "hello world"}]))

    # from lightrag.core.embedder import Embedder
    # from lightrag.core.types import ModelClientType
    # from lightrag.components.data_process import DocumentSplitter, ToEmbeddings
    # from lightrag.core.component import Sequential
    # from lightrag.utils import setup_env  # noqa
    # from lightrag.utils import enable_library_logging

    # enable_library_logging(level="DEBUG")

    # model_kwargs = {
    #     "model": "text-embedding-3-small",
    #     "dimensions": 256,
    #     "encoding_format": "float",
    # }

    # splitter_config = {"split_by": "word", "split_length": 50, "split_overlap": 10}

    # splitter = DocumentSplitter(**splitter_config)
    # embedder = Embedder(
    #     model_client=ModelClientType.OPENAI(), model_kwargs=model_kwargs
    # )
    # embedder_transformer = ToEmbeddings(embedder, batch_size=2)
    # data_transformer = Sequential(splitter, embedder_transformer)

    # print(f"is embedder_transformer: {data_transformer.is_picklable()}")

    # db = LocalDB()
    # db.load([Document(text="hello world"), Document(text="hello world2")])

    # db.transform_data(key="test", transformer=data_transformer)

    # print(f"is db picklable: {db.is_picklable()}")
    # db.pickle_to_file("storage/local_document_db.pkl")
    # db2 = LocalDB.load_from_pickle("storage/local_document_db.pkl")
    # # db.save_state("storage/local_document_db.pkl")
    # # db2 = LocalDB.load_state("storage/local_document_db.pkl")

    # print(db2)
    # print(db2.transformer_setups["test"])
    # print(db.transformer_setups["test"])
    # # db.save_state("storage/local_document_db.pkl")
    # # db2 = LocalDB.load_state("storage/local_document_db.pkl")
    # # print(db2.transformed_items["test"])
