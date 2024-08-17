from types import ModuleType
import pytest
import unittest
from adalflow.utils.lazy_import import safe_import


class TestSafeImport(unittest.TestCase):
    def test_import_installed_package(self):
        module = safe_import("numpy", "Please install math with: pip install numpy")
        self.assertIsInstance(
            module, ModuleType, f"Expected module type, got {type(module)}"
        )
        self.assertTrue(
            hasattr(module, "__version__"),
            "Module 'numpy' should have attribute '__version__'",
        )

    def test_import_nonexistent_package(self):
        with self.assertRaises(ImportError) as cm:
            safe_import(
                "nonexistent_package",
                "Please install nonexistent_package with: pip install nonexistent_package",
            )
        self.assertIn(
            "Please install nonexistent_package with: pip install nonexistent_package",
            str(cm.exception),
            (f"Expected error message not found in {str(cm.exception)}"),
        )


if __name__ == "__main__":
    pytest.main()
