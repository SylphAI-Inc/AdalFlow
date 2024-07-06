import unittest
import pytest
from collections import OrderedDict

from lightrag.core import Sequential, Component


class AddOne(Component):
    def call(self, input: int) -> int:
        return input + 1


class MultiplyByTwo(Component):
    def call(self, input: int) -> int:
        return input * 2


class SequentialTests(unittest.TestCase):

    def setUp(self):
        self.add_one = AddOne()
        self.multiply_by_two = MultiplyByTwo()
        self.seq = Sequential(self.add_one, self.multiply_by_two)

    def test_initialization_positional(self):
        self.assertEqual(len(self.seq), 2)
        self.assertIsInstance(self.seq[0], AddOne)
        self.assertIsInstance(self.seq[1], MultiplyByTwo)

    def test_initialization_ordered_dict(self):
        add_one = AddOne()
        multiply_by_two = MultiplyByTwo()
        components = OrderedDict([("one", add_one), ("two", multiply_by_two)])
        seq = Sequential(components)
        assert len(seq) == 2
        print(seq)
        assert isinstance(seq["one"], AddOne)
        assert isinstance(seq["two"], MultiplyByTwo)

    def test_append_component(self):
        seq = Sequential()
        seq.append(self.add_one)
        self.assertEqual(len(seq), 1)
        self.assertIsInstance(seq[0], AddOne)

    def test_insert_component(self):
        seq = Sequential(self.multiply_by_two)
        seq.insert(0, self.add_one)
        self.assertEqual(len(seq), 2)
        self.assertIsInstance(seq[0], AddOne)
        self.assertIsInstance(seq[1], MultiplyByTwo)

    def test_delete_component(self):
        del self.seq[0]
        self.assertEqual(len(self.seq), 1)
        self.assertIsInstance(self.seq[0], MultiplyByTwo)

    def test_get_component(self):
        self.assertIsInstance(self.seq[0], AddOne)
        self.assertIsInstance(self.seq[1], MultiplyByTwo)

    def test_call(self):
        result = self.seq.call(2)
        self.assertEqual(result, 6)  # (2 + 1) * 2

    def test_add_sequential(self):
        seq1 = Sequential(self.add_one)
        seq2 = Sequential(self.multiply_by_two)
        seq3 = seq1 + seq2
        self.assertEqual(len(seq3), 2)
        self.assertIsInstance(seq3[0], AddOne)
        self.assertIsInstance(seq3[1], MultiplyByTwo)

    def test_iadd_sequential(self):
        seq1 = Sequential(self.add_one)
        seq2 = Sequential(self.multiply_by_two)
        seq1 += seq2
        self.assertEqual(len(seq1), 2)
        self.assertIsInstance(seq1[0], AddOne)
        self.assertIsInstance(seq1[1], MultiplyByTwo)


class Add(Component):
    def call(self, x: int, y: int) -> int:
        return x + y


class Multiply(Component):
    def call(self, x: int, factor: int = 1) -> int:
        return x * factor


class Subtract(Component):
    def call(self, x: int, subtractor: int = 0) -> int:
        return x - subtractor


class TestSequential:

    @pytest.fixture
    def setup_advanced_components(self):
        add = Add()
        multiply = Multiply()
        subtract = Subtract()
        return add, multiply, subtract, Sequential(add, multiply, subtract)

    def test_call_with_single_argument(self):
        add = Add()
        multiply = Multiply()
        seq = Sequential(add, multiply)
        result = seq.call(2, 3)
        assert result == 5  # (2 + 3) * 1

    def test_call_with_multiple_arguments(self, setup_advanced_components):
        add, multiply, subtract, seq = setup_advanced_components
        with pytest.raises(TypeError):
            seq.call(2, 3, 4, 5)  # Too many positional arguments

    def test_call_with_kwargs_only(self, setup_advanced_components):
        add, multiply, subtract, seq = setup_advanced_components
        result = seq.call(x=2, y=3)
        assert result == 5

    def test_call_with_mixed_args_kwargs(self, setup_advanced_components):
        add, multiply, subtract, seq = setup_advanced_components
        result = seq.call(2, y=3)
        assert result == 5

    def test_call_with_only_positional_args(self, setup_advanced_components):
        add, multiply, subtract, seq = setup_advanced_components
        with pytest.raises(TypeError):
            seq.call(2, 3, 4)  # Too many positional arguments

    def test_call_with_partial_kwargs(self, setup_advanced_components):
        add, multiply, subtract, seq = setup_advanced_components
        result = seq.call(2, y=3)
        assert result == 5  # ((2 + 3) * 1) - 1


if __name__ == "__main__":
    unittest.main()
