import unittest
import numpy as np
from adalflow.utils.data import DataLoader

# Assuming DataLoader is imported or defined in the same file


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Set up a simple dataset
        self.dataset = [i for i in range(10)]
        self.dataset_2 = [i for i in range(9)]
        self.dataset_3 = [{"a": i, "b": i * 2} for i in range(11)]
        self.dataset_4 = [{"a": i, "b": i * 2} for i in range(12)]
        self.tuple_dataset = [(i, i * 2) for i in range(10)]

    def test_dataloader_length(self):
        # Test if DataLoader length is correct
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=False)
        self.assertEqual(len(dataloader), 3)
        dataloader_2 = DataLoader(self.dataset_2, batch_size=3, shuffle=False)
        self.assertEqual(len(dataloader_2), 3)
        dataloader_3 = DataLoader(self.dataset_3, batch_size=3, shuffle=False)
        self.assertEqual(len(dataloader_3), 4)
        dataloader_4 = DataLoader(self.dataset_4, batch_size=4, shuffle=False)
        self.assertEqual(len(dataloader_4), 3)

    def test_dataloader_batches(self):
        # Test if DataLoader returns correct batches without shuffle
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=False)
        batches = list(dataloader)
        expected_batches = [
            np.array([0, 1, 2, 3]),
            np.array([4, 5, 6, 7]),
            np.array([8, 9]),
        ]
        self.assertEqual(len(batches), 3)
        for batch, expected in zip(batches, expected_batches):
            np.testing.assert_array_equal(batch, expected)

    def test_dataloader_shuffle(self):
        # Test if DataLoader shuffles the data correctly
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=True)
        batches = list(dataloader)
        self.assertEqual(len(batches), 3)
        all_items = np.concatenate(batches)
        self.assertEqual(sorted(all_items), self.dataset)

    def test_dataloader_with_tuples(self):
        # Test if DataLoader handles datasets with tuples correctly
        dataloader = DataLoader(self.tuple_dataset, batch_size=4, shuffle=False)
        batches = list(dataloader)
        expected_batches = [
            (np.array([0, 1, 2, 3]), np.array([0, 2, 4, 6])),
            (np.array([4, 5, 6, 7]), np.array([8, 10, 12, 14])),
            (np.array([8, 9]), np.array([16, 18])),
        ]
        self.assertEqual(len(batches), 3)
        for batch, expected in zip(batches, expected_batches):
            for b, e in zip(batch, expected):
                np.testing.assert_array_equal(b, e)

    def test_dataloader_stopiteration(self):
        # Test if DataLoader raises StopIteration correctly
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=False)
        dataloader_iter = iter(dataloader)
        for _ in range(len(dataloader)):
            next(dataloader_iter)
        with self.assertRaises(StopIteration):
            next(dataloader_iter)

    def test_dataloader_partial_batch(self):
        # Test if DataLoader handles the last partial batch correctly
        dataloader = DataLoader(self.dataset, batch_size=3, shuffle=False)
        batches = list(dataloader)
        expected_batches = [
            np.array([0, 1, 2]),
            np.array([3, 4, 5]),
            np.array([6, 7, 8]),
            np.array([9]),
        ]
        self.assertEqual(len(batches), 4)
        for batch, expected in zip(batches, expected_batches):
            np.testing.assert_array_equal(batch, expected)

    def test_dataloader_steps_larger_than_length(self):
        # Test if DataLoader handles steps larger than the total length of dataloader
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=False)
        length = len(dataloader)
        dataloader.set_max_steps(length + 1)
        for _ in range(length + 1):
            next(dataloader)


if __name__ == "__main__":
    unittest.main()
