"""Lazy import a module and class."""

import importlib
import logging
from types import ModuleType


from enum import Enum

log = logging.getLogger(__name__)


class OptionalPackages(Enum):
    __doc__ = r"""Enum for optional packages that can be used in the library.

    The package name and error message are defined for each optional package as a tuple.
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
    # modeling library
    TORCH = ("torch", "Please install torch with: pip install torch")

    # search library
    FAISS = (
        "faiss",
        "Please install faiss with: pip install faiss-cpu (or faiss if you use GPU)",
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


def safe_import(module_name: str, install_message: str) -> ModuleType:
    """Safely import a module and raise an ImportError with the install message if the module is not found.

    Mainly used internally to import optional packages only when needed.

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
    try:
        return importlib.import_module(module_name)
    except ImportError:
        raise ImportError(f"{install_message}")
