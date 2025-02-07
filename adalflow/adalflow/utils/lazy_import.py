"""Lazy import a module and class."""

from typing import List, Union
import importlib
import logging
from types import ModuleType

from enum import Enum

log = logging.getLogger(__name__)


class OptionalPackages(Enum):
    __doc__ = r"""Enum for optional packages that can be used in the library.

    The package name and error message are defined for each optional package as a tuple.

    The value of the tuple:
    - The package name (str): The package name to import. Follows the right syntax: such as import azure.identity but the package itself is azure-identity.
      Support a list of package names for related packages. This will be importing a list of packages while safe_import is used.
    - The error message (str): The message to display if the package is not found.

    Example of using multiple related packages:

    .. code-block:: python

        from adalflow.utils.lazy_import import safe_import, OptionalPackages
        import sys

        azure_modules = safe_import(
        OptionalPackages.AZURE.value[0],  # List of package names
        OptionalPackages.AZURE.value[1],  # Error message
        )
        # Manually add each module to sys.modules to make them available globally as if imported normally
        azure_module_names = OptionalPackages.AZURE.value[0]
        for name, module in zip(azure_module_names, azure_modules):
            sys.modules[name] = module

        # Use the modules as if they were imported normally
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    """
    # model sdk
    GROQ = ("groq", "Please install groq with: pip install groq")
    OPENAI = ("openai", "Please install openai with: pip install openai")
    ANTHROPIC = ("anthropic", "Please install anthropic with: pip install anthropic")
    GOOGLE_GENERATIVEAI = (
        "google.generativeai",
        "Please install google-generativeai with: pip install google-generativeai",
    )
    TRANSFORMERS = (
        "transformers",
        "Please install transformers with: pip install transformers",
    )
    COHERE = ("cohere", "Please install cohere with: pip install cohere")
    OLLAMA = ("ollama", "Please install ollama with: pip install ollama")
    # AWS
    BOTO3 = (
        ["boto3", "botocore"],
        "Please install boto3 and botocore with: pip install boto3 botocore",
    )
    # modeling library
    TORCH = ("torch", "Please install torch with: pip install torch")

    # Grouping all Azure-related packages under one entry
    AZURE = (
        [
            "azure.identity",
            "azure.core",
            # "azure.ai-formrecognizer",
            # "azure.ai-textanalytics",
        ],
        "Please install Azure packages with: pip install azure-identity azure-core azure-ai-formrecognizer azure-ai-textanalytics",
    )
    TOGETHER = ("together", "Please install together with: pip install together")
    MISTRAL = ("mistralai", "Please install mistralai with: pip install mistrali")
    FIREWORKS = (
        "fireworks-ai",
        "Please install fireworks-ai with: pip install fireworks-ai",
    )
    # search library
    FAISS = (
        "faiss",
        "Please install faiss with: pip install faiss-cpu (or faiss if you use GPU)",
    )

    LANCEDB = (
        "lancedb",
        "Please install lancedb with: pip install lancedb .",
    )

    # db library
    SQLALCHEMY = (
        "sqlalchemy",
        "Please install sqlalchemy with: pip install sqlalchemy",
    )
    PGVECTOR = (
        "pgvector",
        "Please install pgvector with: pip install pgvector",
    )
    DATASETS = (
        "datasets",
        "Please install datasets with: pip install datasets",
    )
    QDRANT = (
        "qdrant-client",
        "Please install qdrant-client with: pip install qdrant-client",
    )

    def __init__(self, package_name, error_message):
        self.package_name = package_name
        self.error_message = error_message


class LazyImport:
    __doc__ = r"""Lazy import a module and class.

    It is a proxy. The class/func will be created only when the class is instantiated or called.

    .. note::

       Do not subclass a lazy imported class.

    Mainly internal library use to import optional packages only when needed.

    Args:
        import_path (str): The import path of the module and class, eg. "adalflow.components.model_client.openai_client.OpenAIClient".
        optional_package (OptionalPackages): The optional package to import, it helps define the package name and error message.
    """

    def __init__(
        self, import_path: str, optional_package: OptionalPackages, *args, **kwargs
    ):
        if args or kwargs:
            raise TypeError(
                "LazyImport does not support subclassing or additional arguments. "
                "Import the class directly from its specific module instead. For example, "
                "from adalflow.components.model_client.cohere_client import CohereAPIClient"
                "instead of using the lazy import with: from adalflow.components.model_client import CohereAPIClient"
            )
        self.import_path = import_path
        self.optional_package = optional_package
        self.module = None
        self.class_ = None

    def load_class(self):
        log.debug(f"Loading class: {self.import_path}")
        if self.module is None:
            try:
                components = self.import_path.split(".")
                module_path = ".".join(components[:-1])
                class_name = components[-1]
            except Exception as e:
                log.error(f"Error parsing import path: {e}")
            try:
                self.module = importlib.import_module(module_path)
                self.class_ = getattr(self.module, class_name)
                return self.class_
            except ImportError:
                log.info(f"Optional module not installed: {self.import_path}")
                raise ImportError(f"{self.optional_package.error_message}")

    def __getattr__(self, name):
        self.load_class()
        return getattr(self.class_, name)

    def __call__(self, *args, **kwargs):
        """Create the class instance."""
        self.load_class()
        log.debug(f"Creating class instance: {self.class_}")
        # normal class initialization
        return self.class_(*args, **kwargs)


def safe_import(
    module_names: Union[str, List[str]], install_message: str
) -> Union[ModuleType, List[ModuleType]]:
    """Safely import a module and raise an ImportError with the install message if the module is not found.

    Handles importing of multiple related packages.

    Args:
        module_names (list or str): The package name(s) to import.
        install_message (str): The message to display if import fails.

    Returns:
        ModuleType: The imported module.

    Raises:
        ImportError: If any of the packages are not found.


    Example:

    1. Tests

    .. code-block:: python

        try:
            numpy = safe_import("numpy", "Please install numpy with: pip install numpy")
            print(numpy.__version__)
        except ImportError as e:
            print(e)

    When numpy is not installed, it will raise an ImportError with the install message.
    When numpy is installed, it will print the numpy version.

    2. Use it to delay the import of optional packages in the library.

    .. code-block:: python

        from adalflow.utils.lazy_import import safe_import, OptionalPackages

        numpy = safe_import(OptionalPackages.NUMPY.value[0], OptionalPackages.NUMPY.value[1])

    """
    if isinstance(module_names, str):
        module_names = [module_names]

    return_modules = []
    for module_name in module_names:
        try:
            return_modules.append(importlib.import_module(module_name))
        except ImportError:
            raise ImportError(f"{install_message}")

    return return_modules[0] if len(return_modules) == 1 else return_modules
