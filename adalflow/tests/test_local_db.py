import unittest
import os
import shutil
from typing import Any, List, Dict

# Assuming your LocalDB implementation is in local_db.py
# If it's named differently or in a different folder, adjust the import.
from adalflow.core import LocalDB, Component


class MockTransformer(Component):
    """
    A simple mock transformer that just appends '_transformed' to each item
    if it's a string. Otherwise returns the item as-is.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, items: List[Any]) -> List[Any]:
        transformed_items = []
        for item in items:
            if isinstance(item, str):
                transformed_items.append(item + "_transformed")
            else:
                transformed_items.append(item)
        return transformed_items


class MockTransformer2(Component):
    """
    Another mock transformer that converts strings to uppercase.
    """

    def __call__(self, items: List[Any]) -> List[Any]:
        transformed = []
        for item in items:
            if isinstance(item, str):
                transformed.append(item.upper())
            else:
                transformed.append(item)
        return transformed

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MockTransformer2":
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        return {}


class TestLocalDB(unittest.TestCase):

    def setUp(self):
        """
        Create a fresh LocalDB instance for each test.
        """
        self.db = LocalDB(name="test_db")

    def tearDown(self):
        """
        Clean up any generated files or directories.
        """
        # Remove any saved pickle file if it exists
        if self.db.index_path and os.path.exists(self.db.index_path):
            os.remove(self.db.index_path)
        # Additionally, remove any leftover directories
        if os.path.exists("local_db"):
            shutil.rmtree("local_db")

    def test_initial_state(self):
        """
        Test the initial state of a newly instantiated LocalDB.
        """
        self.assertEqual(self.db.name, "test_db")
        self.assertEqual(self.db.items, [])
        self.assertEqual(self.db.transformed_items, {})
        self.assertEqual(self.db.transformer_setups, {})
        self.assertEqual(self.db.mapper_setups, {})
        self.assertEqual(self.db.length, 0)

    def test_load_items(self):
        """
        Test loading items into the LocalDB.
        """
        data = ["hello", "world"]
        self.db.load(data)
        self.assertEqual(self.db.items, data)
        self.assertEqual(self.db.length, 2)

    def test_extend_items_without_transform(self):
        """
        Test extending the LocalDB without applying transformations.
        """
        initial_data = ["hello", "world"]
        self.db.reset()
        self.db.load(initial_data)
        extended_data = ["this", "is", "extended"]
        self.db.extend(extended_data, apply_transformer=False)

        self.assertEqual(self.db.items, initial_data + extended_data)
        self.assertEqual(self.db.length, 5)

    def test_extend_items_with_transform(self):
        """
        Test extending the LocalDB with transformations.
        """
        # Register a transformer
        transformer = MockTransformer()
        self.db.register_transformer(transformer=transformer, key="mock_transformer")

        self.db.load(["first", "second"])  # not transformed
        self.db.extend(["third", "fourth"], apply_transformer=True)

        self.assertIn("mock_transformer", self.db.transformed_items)
        # Check that new items are transformed
        expected_transformed = [
            "third_transformed",
            "fourth_transformed",
        ]
        self.assertEqual(
            self.db.transformed_items["mock_transformer"], expected_transformed
        )

    def test_register_transformer_with_map_fn(self):
        """
        Test registering a transformer with a map function.
        """

        # Example map function: reverse each string before transformation
        def reverse_str(item: str) -> str:
            return item[::-1]

        transformer = MockTransformer()
        self.db.register_transformer(
            transformer=transformer, key="reverse_transformer", map_fn=reverse_str
        )

        self.db.load(["abc", "def"])
        self.db.extend(["ghi"], apply_transformer=True)

        # The reversed items will be passed to the transformer
        # So "abc" -> "cba" -> "cba_transformed"
        expected_transformed = ["ihg_transformed"]
        self.assertEqual(
            self.db.transformed_items["reverse_transformer"], expected_transformed
        )

    # def test_transform_method_overloads(self):
    #     """
    #     Test the different overloads of transform() method.
    #     """
    #     transformer = MockTransformer()

    #     # 1) Register a transformer, then apply by key
    #     self.db.register_transformer(transformer, key="my_transformer")
    #     self.db.load(["one", "two"])
    #     self.db.transform("my_transformer")  # Overload 1
    #     self.assertEqual(
    #         self.db.transformed_items["my_transformer"],
    #         ["one_transformed", "two_transformed"],
    #     )

    #     # 2) Directly call transform() with a new transformer
    #     self.db.transform(MockTransformer2(), key="upper_transformer")
    #     self.assertEqual(self.db.transformed_items["upper_transformer"], ["ONE", "TWO"])

    # def test_delete_items(self):
    #     """
    #     Test deleting items by index, ensuring transformed items are also updated.
    #     """
    #     transformer = MockTransformer()
    #     self.db.register_transformer(transformer, key="mock")
    #     self.db.load(["a", "b", "c"])

    #     # Transform the entire dataset
    #     self.db.transform("mock")
    #     self.assertEqual(
    #         self.db.transformed_items["mock"],
    #         ["a_transformed", "b_transformed", "c_transformed"],
    #     )

    #     # Delete the second item
    #     self.db.delete(index=1)  # remove 'b'
    #     self.assertEqual(self.db.items, ["a", "c"])
    #     self.assertEqual(
    #         self.db.transformed_items["mock"], ["a_transformed", "c_transformed"]
    #     )

    # def test_add_item_without_transformer(self):
    #     """
    #     Test adding a single item when there's no transformer.
    #     """
    #     self.db.load(["existing_item"])
    #     self.db.add("new_item")
    #     self.assertEqual(self.db.items, ["existing_item", "new_item"])

    # def test_add_item_with_transformer(self):
    #     """
    #     Test adding a single item at a specific index with an active transformer.
    #     """
    #     transformer = MockTransformer()
    #     self.db.register_transformer(transformer, key="mock")
    #     # Load initial data
    #     self.db.load(["hello", "world"])
    #     # Transform initial data
    #     self.db.transform("mock")
    #     # Insert at index 1
    #     self.db.add("inserted", index=1)
    #     self.assertEqual(self.db.items, ["hello", "inserted", "world"])

    #     # The transformed data should also have 'inserted_transformed' at index 1
    #     self.assertEqual(
    #         self.db.transformed_items["mock"],
    #         ["hello_transformed", "inserted_transformed", "world_transformed"],
    #     )

    def test_reset_db(self):
        """
        Test resetting the DB, ensuring all items and transformers are cleared.
        """
        transformer = MockTransformer()
        self.db.reset()
        self.db.register_transformer(transformer=transformer, key="mock")
        self.db.load(["reset_me"])
        self.db.transform(key="mock")

        self.db.reset()
        self.assertEqual(self.db.items, [])
        self.assertEqual(self.db.transformer_setups, {})
        self.assertEqual(self.db.mapper_setups, {})

    # def test_fetch_items(self):
    #     """
    #     Test fetching items with a condition.
    #     """
    #     self.db.load(["apple", "banana", "pear", "pineapple"])
    #     result = self.db.fetch_items(lambda x: "apple" in x)
    #     self.assertEqual(result, ["apple", "pineapple"])

    # def test_fetch_transformed_items(self):
    #     """
    #     Test fetching transformed items with a condition.
    #     """
    #     transformer = MockTransformer()
    #     self.db.register_transformer(transformer, key="mock")
    #     self.db.load(["apple", "banana", "pear", "pineapple"])
    #     self.db.transform("mock")

    #     # Condition for items that contain "apple"
    #     result = self.db.fetch_transformed_items("mock", lambda x: "apple" in x)
    #     # Expect: ["apple_transformed", "pineapple_transformed"]
    #     self.assertEqual(result, ["apple_transformed", "pineapple_transformed"])

    def test_save_and_load_state(self):
        """
        Test saving the DB state to a pickle file and re-loading it.
        Checks that attributes match.
        """
        transformer = MockTransformer()
        self.db.register_transformer(transformer=transformer, key="mock")
        self.db.load(["save", "load"])
        self.db.transform(key="mock")

        save_path = "./test_local_db.pkl"
        self.db.save_state(save_path)

        # # Load from pickle
        loaded_db = LocalDB.load_state(save_path)

        self.db.to_dict() == loaded_db.to_dict()

        self.assertIsNotNone(loaded_db)
        self.assertEqual(self.db.name, loaded_db.name)
        self.assertEqual(self.db.items, loaded_db.items)
        self.assertEqual(self.db.transformed_items, loaded_db.transformed_items)
        self.assertIn("mock", loaded_db.transformer_setups)

        # Cleanup
        if os.path.exists(save_path):
            os.remove(save_path)

    def test_index_path_persistence(self):
        """
        Test if index_path is set correctly during save and load.
        """
        self.db.load(["index_path_item"])
        filepath = "./test_local_db.pkl"
        self.db.save_state(filepath)

        self.assertEqual(self.db.index_path, filepath)

        loaded_db = LocalDB.load_state(filepath)
        self.assertEqual(loaded_db.index_path, filepath)

        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    unittest.main()
