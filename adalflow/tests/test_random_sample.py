import unittest
from typing import TypeVar

# Assuming the random_sample function is defined here or imported
T_co = TypeVar("T_co", covariant=True)


from adalflow.core.functional import random_sample


class TestRandomSample(unittest.TestCase):

    def setUp(self):
        """Set up a common dataset for testing."""
        self.dataset = [1, 2, 3, 4, 5]

    def test_random_sample_no_replacement(self):
        """Test random sampling without replacement."""
        result = random_sample(self.dataset, 3, replace=False)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(item in self.dataset for item in result))
        self.assertEqual(len(set(result)), 3)  # No duplicates

    def test_random_sample_with_replacement(self):
        """Test random sampling with replacement."""
        result = random_sample(self.dataset, 10, replace=True)
        self.assertEqual(len(result), 10)
        self.assertTrue(all(item in self.dataset for item in result))

    def test_weighted_sampling_all_zero_weights(self):
        """Test weighted sampling with all zero weights."""
        result = random_sample(self.dataset, 3, replace=False, weights=[0, 0, 0, 0, 0])
        self.assertEqual(len(result), 3)
        self.assertTrue(all(item in self.dataset for item in result))

    def test_weights_none(self):
        """Test weighted sampling with None weights."""
        result = random_sample(self.dataset, 3, replace=False, weights=None)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(item in self.dataset for item in result))

    def test_weighted_sampling_normal_weights(self):
        """Test weighted sampling with normal weights."""
        weights = [0.1, 0.2, 0.3, 0.4, 0.0]
        result = random_sample(self.dataset, 3, replace=False, weights=weights)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(item in self.dataset for item in result))


if __name__ == "__main__":
    unittest.main()
