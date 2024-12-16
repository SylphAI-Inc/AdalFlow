import unittest

# Assuming `Component` and `ComponentList` are defined in a module named `adalflow.core`
from adalflow.core import Component, ComponentList


class MockComponent(Component):
    """A mock component used for testing purposes."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def __repr__(self):
        return f"MockComponent({self.value})"


class TestComponentList(unittest.TestCase):
    def setUp(self):
        """Create some mock components for testing."""
        self.c1 = MockComponent(1)
        self.c2 = MockComponent(2)
        self.c3 = MockComponent(3)

    def test_initialization(self):
        """Test that ComponentList initializes correctly with components."""
        cl = ComponentList([self.c1, self.c2])
        self.assertEqual(len(cl), 2)
        self.assertIs(cl[0], self.c1)
        self.assertIs(cl[1], self.c2)

    def test_append(self):
        """Test appending a new component to the list."""
        cl = ComponentList([self.c1])
        cl.append(self.c2)
        self.assertEqual(len(cl), 2)
        self.assertIs(cl[1], self.c2)

    def test_extend(self):
        """Test extending the list with multiple components."""
        cl = ComponentList([self.c1])
        cl.extend([self.c2, self.c3])
        self.assertEqual(len(cl), 3)
        self.assertIs(cl[1], self.c2)
        self.assertIs(cl[2], self.c3)

    def test_indexing(self):
        """Test retrieving components by index."""
        cl = ComponentList([self.c1, self.c2, self.c3])
        self.assertIs(cl[0], self.c1)
        self.assertIs(cl[2], self.c3)

    def test_slicing(self):
        """Test slicing the list of components."""
        cl = ComponentList([self.c1, self.c2, self.c3])
        sliced = cl[1:]
        self.assertEqual(len(sliced), 2)
        self.assertIs(sliced[0], self.c2)
        self.assertIs(sliced[1], self.c3)

    def test_insert(self):
        """Test inserting a component at a specific index."""
        cl = ComponentList([self.c1, self.c3])
        cl.insert(1, self.c2)
        self.assertEqual(len(cl), 3)
        self.assertIs(cl[1], self.c2)

    def test_pop(self):
        """Test removing and returning a component."""
        cl = ComponentList([self.c1, self.c2, self.c3])
        component = cl.pop(1)
        self.assertIs(component, self.c2)
        self.assertEqual(len(cl), 2)

    def test_delitem(self):
        """Test deleting components by index and slice."""
        cl = ComponentList([self.c1, self.c2, self.c3])
        del cl[1]
        self.assertEqual(len(cl), 2)
        self.assertIs(cl[0], self.c1)
        self.assertIs(cl[1], self.c3)

        cl = ComponentList([self.c1, self.c2, self.c3])
        del cl[1:]
        self.assertEqual(len(cl), 1)
        self.assertIs(cl[0], self.c1)

    def test_add(self):
        """Test adding two ComponentLists."""
        cl1 = ComponentList([self.c1])
        cl2 = ComponentList([self.c2, self.c3])
        cl3 = cl1 + cl2
        self.assertEqual(len(cl3), 3)
        self.assertIs(cl3[0], self.c1)
        self.assertIs(cl3[1], self.c2)
        self.assertIs(cl3[2], self.c3)

    def test_iadd(self):
        """Test adding components using the += operator."""
        cl = ComponentList([self.c1])
        cl += [self.c2, self.c3]
        self.assertEqual(len(cl), 3)
        self.assertIs(cl[1], self.c2)
        self.assertIs(cl[2], self.c3)

    def test_repr(self):
        """Test the custom __repr__ implementation."""
        cl = ComponentList([MockComponent(1), MockComponent(1), MockComponent(2)])
        expected = (
            "ComponentList(\n  (0-1): 2 x MockComponent(1)\n  (2): MockComponent(2)\n)"
        )
        self.assertEqual(repr(cl), expected)

    def test_len(self):
        """Test the length of the ComponentList."""
        cl = ComponentList([self.c1, self.c2])
        self.assertEqual(len(cl), 2)

    def test_iter(self):
        """Test iterating over the components."""
        cl = ComponentList([self.c1, self.c2, self.c3])
        components = list(iter(cl))
        self.assertEqual(components, [self.c1, self.c2, self.c3])


if __name__ == "__main__":
    unittest.main()
