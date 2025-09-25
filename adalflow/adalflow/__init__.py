__version__ = "1.1.3" # DO NOT EDIT THIS LINE MANUALLY

from adalflow.core.component import (
    Component,
    DataComponent,
    FuncDataComponent,
    func_to_data_component,
)
from adalflow.core.container import Sequential, ComponentList
from adalflow.core.base_data_class import DataClass, DataClassFormatType, required_field

from adalflow.optim.grad_component import GradComponent
from adalflow.core.generator import Generator

from adalflow.core.types import (
    GeneratorOutput,
    EmbedderOutput,
    RetrieverOutput,
    Document,
)
from adalflow.core.model_client import ModelClient
from adalflow.core.embedder import Embedder, BatchEmbedder

# parser
from adalflow.core.string_parser import (
    YamlParser,
    JsonParser,
    IntParser,
    FloatParser,
    ListParser,
    BooleanParser,
)
from adalflow.core.retriever import Retriever
from adalflow.components.output_parsers import (
    YamlOutputParser,
    JsonOutputParser,
    ListOutputParser,
)
from adalflow.components.output_parsers.dataclass_parser import DataClassParser

from adalflow.core.prompt_builder import Prompt

# optimization
from adalflow.optim import (
    Optimizer,
    DemoOptimizer,
    TextOptimizer,
    Parameter,
    AdalComponent,
    Trainer,
    BootstrapFewShot,
    TGDOptimizer,
    EvalFnToTextLoss,
    LLMAsTextLoss,
)

from adalflow.optim.types import ParameterType
from adalflow.utils import setup_env, get_logger

from adalflow.components.model_client import (
    OpenAIClient,
    GoogleGenAIClient,
    GroqAPIClient,
    OllamaClient,
    TransformersClient,
    CohereAPIClient,
    BedrockAPIClient,
    DeepSeekClient,
    TogetherClient,
    AnthropicAPIClient,
)

# data pipeline
from adalflow.components.data_process.text_splitter import TextSplitter

# agents - Import separately to avoid circular imports
from adalflow.components.agent import Agent, Runner, ReActAgent

__all__ = [
    "Component",
    "DataComponent",
    "FuncDataComponent",
    "func_to_data_component",
    # dataclass
    "DataClass",
    "DataClassFormatType",
    "required_field",
    # Container
    "Sequential",
    "ComponentList",
    # Grad Component
    "GradComponent",
    # Functional Component
    "ModelClient",
    "Generator",
    "Embedder",
    "BatchEmbedder",
    "Retriever",
    "Parameter",
    "AdalComponent",
    "Trainer",
    "BootstrapFewShot",
    "TGDOptimizer",
    "EvalFnToTextLoss",
    "LLMAsTextLoss",
    "setup_env",
    "get_logger",
    "Prompt",
    # Parsers
    "YamlParser",
    "JsonParser",
    "IntParser",
    "FloatParser",
    "ListParser",
    "BooleanParser",
    "Parser",
    "FuncParser",
    # Output Parsers with dataclass formatting
    "YamlOutputParser",
    "JsonOutputParser",
    "ListOutputParser",
    "DataClassParser",
    # Data Pipeline
    "TextSplitter",
    "ToEmbeddings",
    # Types
    "GeneratorOutput",
    "EmbedderOutput",
    "RetrieverOutput",
    "Document",
    # Optimizer types
    "Optimizer",
    "DemoOptimizer",
    "TextOptimizer",
    # parameter types
    "ParameterType",
    # model clients
    "OpenAIClient",
    "GoogleGenAIClient",
    "GroqAPIClient",
    "DeepSeekClient",
    "OllamaClient",
    "TransformersClient",
    "AnthropicAPIClient",
    "CohereAPIClient",
    "BedrockAPIClient",
    "TogetherClient",
    "AnthropicAPIClient",
    # Agent
    "ReActAgent",
    "Agent",
    "Runner",
]
