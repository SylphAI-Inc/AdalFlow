"""LocalDB to perform in-memory storage and data persistence(pickle or any filesystem) for data models like documents and dialogturn."""

from typing import List, Optional, Callable, Dict, Any, TypeVar, Generic, overload
import logging
import os
from dataclasses import field, dataclass
import pickle


from lightrag.core.component import Component
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
    4. add: Add a single item to the db. ``db.add({"text": "hello world4"})``.
       In default, the transformer is applied and the transformed data is added.
       Unless the transformed data keeps the same length as the original data, the insert operation does not mean insert after the last item.
    5. delete: Remove items by index. ``db.delete([0])``.
    6. reset: Remove all items. ``db.reset()``, including transformed_items and transformer_setups,and mapper_setups.

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

            db.transform(transformer, key="test", map_fn=map_fn)

    Data persistence:
    1. Save the state of the db to a pickle file.

    .. code-block:: python

            db.save_state("storage/local_item_db.pkl")

    2. Load the state of the db from a pickle file.

    .. code-block:: python

            db2 = LocalDB.load_state("storage/local_item_db.pkl")

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
        mapper_setups (Dict[str, Callable[[T], Any]], optional): Map function setup by key. Defaults to {}.
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
    mapper_setups: Dict[str, Callable[[T], Any]] = field(
        default_factory=dict, metadata={"description": "Map function setup by key"}
    )

    @property
    def length(self):
        return len(self.items)

    def get_transformer_keys(self) -> List[str]:
        return list(self.transformed_items.keys())

    def get_transformed_data(self, key: str) -> List[U]:
        """Get the transformed items by key."""
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
            self.mapper_setups[key] = map_fn
        return key

    @overload
    def transform(self, key: str) -> str:
        """Apply the transformer by key to the data."""
        ...

    @overload
    def transform(
        self,
        transformer: Component,
        key: Optional[str] = None,
        map_fn: Optional[Callable[[T], Any]] = None,
    ) -> str:
        """Register and apply the transformer to the data."""
        ...

    def transform(
        self,
        transformer: Optional[Component] = None,
        key: Optional[str] = None,
        map_fn: Optional[Callable[[T], Any]] = None,
    ) -> str:
        """The main method to apply the transformer to the data in two ways:
        1. Apply the transformer by key to the data using ``transform(key="test")``.
        2. Register and apply the transformer to the data using ``transform(transformer, key="test")``.

        Args:
            transformer (Optional[Component], optional): The transformer to use. Defaults to None.
            key (Optional[str], optional): The key to use for the transformer. Defaults to None.
            map_fn (Optional[Callable[[T], Any]], optional): The map function to use. Defaults to None.

        Returns:
            str: The key used for the transformation, from which the transformed data can be accessed.
        """
        key_to_use = key
        if transformer:
            key = self.register_transformer(transformer, key, map_fn)
            key_to_use = key
        if key_to_use is None:
            raise ValueError("Key must be provided.")

        if map_fn is not None:
            items_to_use = [map_fn(item) for item in self.items]
        else:
            items_to_use = self.items.copy()

        transformer_to_use = self.transformer_setups[key_to_use]
        self.transformed_items[key_to_use] = transformer_to_use(items_to_use)
        return key_to_use

    def load(self, items: List[Any]):
        """Load the db with new items.

        Args:
            items (List[Any]): The items to load.

        Examples:

        .. code-block:: python

            db = LocalDB()
            db.load([{"text": "hello world"}, {"text": "hello world2"}])
        """
        self.items = items

    def extend(self, items: List[Any], apply_transformer: bool = True):
        """Extend the db with new items."""
        self.items.extend(items)
        if apply_transformer:
            for key, transformer in self.transformer_setups.items():
                # check if there was a map function registered
                transformed_items = []
                if key in self.mapper_setups:
                    map_fn = self.mapper_setups[key]
                    transformed_items = transformer([map_fn(doc) for doc in items])
                else:
                    transformed_items = transformer(items)
                self.transformed_items[key].extend(transformed_items)

        self.items.extend(items)

    def delete(self, index: Optional[int] = None, remove_transformed: bool = True):
        """Remove items by index or pop the last item. Optionally remove the transformed data as well.

        Assume the transformed item has the same index as the original item. Might not always be the case.

        Args:
            index (Optional[int], optional): The index to remove. Defaults to None.
            remove_transformed (bool, optional): Whether to remove the transformed data as well. Defaults to True.
        """
        if remove_transformed:
            for key in self.transformed_items.keys():
                self.transformed_items[key].pop(index)
        self.items.pop(index)

    def add(
        self, item: Any, index: Optional[int] = None, apply_transformer: bool = True
    ):
        """Add a single item by index or append to the end. Optionally apply the transformer.

        .. note::
            The item will be transformed using the registered transformer.
            Only if the transformed data keeps the same length as the original data, the ``insert`` operation will work correctly.

        Args:
            item (Any): The item to add.
            index (int, optional): The index to add the item at. Defaults to None.
            When None, the item is appended to the end.
            apply_transformer (bool, optional): Whether to apply the transformer to the item. Defaults to True.
        """
        transformed_items: Dict[str, List] = {}
        if apply_transformer:
            for key, transformer in self.transformer_setups.items():
                transformed_docs = []
                map_fn = self.mapper_setups.get(key, None)
                if map_fn is not None:
                    transformed_docs = transformer([map_fn(item)])
                else:
                    transformed_docs = transformer([item])
                transformed_items[key] = transformed_docs

        if index is not None:
            self.items.insert(index, item)
            for key, transformed_docs in transformed_items.items():
                for doc in transformed_docs:
                    self.transformed_items[key].insert(index, doc)
        else:
            self.items.append(item)
            for key, transformed_docs in transformed_items.items():
                self.transformed_items[key].extend(transformed_docs)

    def fetch_items(self, condition: Callable[[T], bool]) -> List[T]:
        """Fetch items with a condition."""
        return [item for item in self.items if condition(item)]

    def fetch_transformed_items(
        self, key: str, condition: Callable[[U], bool]
    ) -> List[U]:
        """Fetch transformed items with a condition."""
        return [item for item in self.transformed_items[key] if condition(item)]

    def reset(self):
        r"""Reset all attributes to empty."""
        self.mapped_items = {}
        self.transformer_setups = {}
        self.mapper_setups = {}
        self.items = []

    def save_state(self, filepath: str):
        """Save the current state (attributes) of the DB using pickle.

        Note:
            The transformer setups will be lost when pickling. As it might not be picklable.
        """
        filepath = filepath or "storage/local_item_db.pkl"
        file_dir = os.path.dirname(filepath)
        if file_dir and file_dir != "":
            os.makedirs(file_dir, exist_ok=True)

        with open(filepath, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_state(cls, filepath: str = None) -> "LocalDB":
        """Load the state of the DB from a pickle file."""
        filepath = filepath or "storage/local_item_db.pkl"
        with open(filepath, "rb") as file:
            return pickle.load(file)

    def __getstate__(self):
        """Special handling of the components in pickling."""
        state = self.__dict__.copy()
        _transformer_files = {}
        _transformer_type_names = {}

        for key, transformer in self.transformer_setups.items():
            _transformer_files[key] = transformer.to_dict()
            _transformer_type_names[key] = transformer.__class__.__name__
        state["transformer_setups"] = {}
        state["_transformer_files"] = _transformer_files
        state["_transformer_type_names"] = _transformer_type_names
        return state

    def __setstate__(self, state):
        """Restore state with special handling of the components."""
        _transformer_files = state.pop("_transformer_files")
        _transformer_type_names = state.pop("_transformer_type_names")
        self.__dict__.update(state)
        for key, transformer_file in _transformer_files.items():
            class_type = (
                EntityMapping.get(_transformer_type_names[key])
                or globals()[_transformer_type_names[key]]
            )
            self.transformer_setups[key] = class_type.from_dict(transformer_file)
