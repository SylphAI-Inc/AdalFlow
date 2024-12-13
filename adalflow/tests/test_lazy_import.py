from types import ModuleType
import pytest
import unittest
from unittest.mock import MagicMock, patch
from adalflow.utils.lazy_import import safe_import, OptionalPackages


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

    @patch("importlib.import_module")
    def test_import_with_multiple_packages_success(self, mock_import):
        """Test that safe_import successfully imports a module if all modules exist"""
        # Set up the mock to return a MagicMock to simulate successful imports
        mock_import.side_effect = [MagicMock(name="successful_import")]

        packages = ["math"]
        imported_module = safe_import(packages, "Please install the required packages.")

        self.assertTrue(
            hasattr(imported_module, "sqrt"), "math module did not load correctly."
        )

    @patch("importlib.import_module")
    def test_import_with_multiple_packages_failure(self, mock_import):
        """Test that safe_import raises ImportError when any of the modules in the list fail to import"""
        # Set up the mock to raise ImportError for the first package
        mock_import.side_effect = ImportError

        packages = ["non_existent_module_1", "math"]
        with self.assertRaises(ImportError) as context:
            safe_import(packages, "Please install the required packages.")

        self.assertIn(
            "Please install the required packages",
            str(context.exception),
            "Expected ImportError message does not match.",
        )

    ############################################################################################################
    # For AWS bedrock_client
    ############################################################################################################
    @patch("importlib.import_module")
    def test_successful_import_boto3(self, mock_import):
        """Test that safe_import successfully imports boto3 if installed"""
        # Simulate a successful import of boto3 by returning a MagicMock object
        mock_import.side_effect = lambda name: (
            MagicMock(name="boto3") if name == "boto3" else ImportError
        )

        module_name = OptionalPackages.BOTO3.value[0]
        imported_module = safe_import(module_name, OptionalPackages.BOTO3.value[1])

        # Assert that the mock was called with 'boto3' and the imported module is a MagicMock instance
        mock_import.assert_called_with("boto3")
        self.assertIsInstance(
            imported_module, MagicMock, "boto3 module did not load correctly."
        )

    @patch("importlib.import_module", side_effect=ImportError)
    def test_failed_import_boto3(self, mock_import):
        """Test that safe_import raises an ImportError when boto3 is not installed"""
        module_name = OptionalPackages.BOTO3.value[0]
        with self.assertRaises(ImportError) as context:
            safe_import(module_name, OptionalPackages.BOTO3.value[1])

        self.assertIn(
            "Please install boto3 with: pip install boto3",
            str(context.exception),
            "Expected ImportError message for boto3 does not match.",
        )

    ############################################################################################################
    # For Azure ai model client
    ############################################################################################################
    @patch("importlib.import_module")
    def test_successful_import_azure(self, mock_import):
        """Test that safe_import successfully imports all Azure packages"""
        # Simulate successful imports for each Azure package by returning a MagicMock for each
        mock_import.side_effect = lambda name: MagicMock(name=name)

        module_names = OptionalPackages.AZURE.value[0]
        imported_modules = safe_import(module_names, OptionalPackages.AZURE.value[1])

        # Ensure that all azure modules were attempted to be imported
        for module_name in module_names:
            mock_import.assert_any_call(module_name)

        # Verify that the imported_modules is a list and contains each module as a MagicMock
        self.assertIsInstance(imported_modules, list, "Expected a list of modules.")
        self.assertEqual(
            len(imported_modules),
            len(module_names),
            "Not all Azure modules were imported.",
        )
        for imported_module, module_name in zip(imported_modules, module_names):
            self.assertIsInstance(
                imported_module, MagicMock, f"{module_name} did not load correctly."
            )

    @patch("importlib.import_module")
    def test_failed_import_azure(self, mock_import):
        """Test that safe_import raises ImportError when any Azure package is not installed"""
        # Set up the mock to raise ImportError for all Azure packages to simulate a missing package
        mock_import.side_effect = ImportError

        module_names = OptionalPackages.AZURE.value[0]
        with self.assertRaises(ImportError) as context:
            safe_import(module_names, OptionalPackages.AZURE.value[1])

        self.assertIn(
            "Please install Azure packages with: pip install azure-identity azure-core azure-ai-formrecognizer azure-ai-textanalytics",
            str(context.exception),
            "Expected ImportError message for Azure packages does not match.",
        )


if __name__ == "__main__":
    pytest.main()
