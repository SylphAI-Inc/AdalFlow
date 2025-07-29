from .base_data_class import DataClass, required_field, DataClassFormatType

from .component import Component, DataComponent, func_to_data_component
from .container import Sequential, ComponentList
from .db import LocalDB
from .default_prompt_template import DEFAULT_ADALFLOW_SYSTEM_PROMPT
from .embedder import Embedder, BatchEmbedder
from .generator import Generator, BackwardEngine
from .model_client import ModelClient
from .string_parser import (
    YamlParser,
    JsonParser,
    IntParser,
    FloatParser,
    ListParser,
    BooleanParser,
)


# from .multirunner import MultiRunner

from .prompt_builder import Prompt

from .retriever import Retriever
from .tokenizer import Tokenizer


from .types import (
    ModelType,
    ModelClientType,
    get_model_args,
    Embedding,
    Usage,
    TokenLogProb,
    EmbedderOutput,
    EmbedderInputType,
    EmbedderOutputType,
    BatchEmbedderInputType,
    BatchEmbedderOutputType,
    GeneratorOutput,
    GeneratorOutputType,
    Document,
    RetrieverQueryType,
    RetrieverStrQueryType,
    RetrieverQueriesType,
    RetrieverStrQueriesType,
    RetrieverDocumentType,
    RetrieverStrDocumentType,
    RetrieverDocumentsType,
    RetrieverOutput,
    RetrieverOutputType,
    UserQuery,
    AssistantResponse,
    DialogTurn,
    Conversation,
)

# Conditional import for MCP tools (requires Python >= 3.10)
try:
    from .mcp_tool import MCPFunctionTool, MCPToolManager
    _MCP_AVAILABLE = True
except ImportError:
    MCPFunctionTool = None
    MCPToolManager = None
    _MCP_AVAILABLE = False

from adalflow.utils.registry import EntityMapping

__all__ = [
    "LocalDB",
    "Component",
    "DataComponent",
    "func_to_data_component",
    "Sequential",
    "ComponentList",
    "DataClass",
    "DataClassFormatType",
    "required_field",
    "Generator",
    "BackwardEngine",
    "Prompt",
    "DEFAULT_ADALFLOW_SYSTEM_PROMPT",
    # "Parameter",
    "required_field",
    "ModelClient",
    "Embedder",
    "BatchEmbedder",
    "Retriever",
    "GeneratorOutput",
    "GeneratorOutputType",
    "ModelType",
    "ModelClientType",
    "get_model_args",
    "Embedding",
    "Usage",
    "TokenLogProb",
    "EmbedderOutput",
    "EmbedderInputType",
    "EmbedderOutputType",
    "BatchEmbedderInputType",
    "BatchEmbedderOutputType",
    "Document",
    "RetrieverQueryType",
    "RetrieverStrQueryType",
    "RetrieverQueriesType",
    "RetrieverStrQueriesType",
    "RetrieverDocumentType",
    "RetrieverStrDocumentType",
    "RetrieverDocumentsType",
    "RetrieverOutput",
    "RetrieverOutputType",
    "UserQuery",
    "AssistantResponse",
    "DialogTurn",
    "Conversation",
    "Tokenizer",
    # Parsers
    "YamlParser",
    "JsonParser",
    "IntParser",
    "FloatParser",
    "ListParser",
    "BooleanParser",
]

# Add MCP tools to __all__ only if available
if _MCP_AVAILABLE:
    __all__.extend(["MCPFunctionTool", "MCPToolManager"])

for name in __all__:
    entity = globals().get(name)
    if entity is not None:
        EntityMapping.register(name, entity)
