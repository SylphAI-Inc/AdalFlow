__version__ = "0.2.0"

from adalflow.core.component import Component, fun_to_component
from adalflow.core.container import Sequential
from adalflow.core.base_data_class import DataClass

from adalflow.optim.grad_component import GradComponent
from adalflow.core.generator import Generator


from adalflow.core.types import GeneratorOutput, EmbedderOutput, RetrieverOutput
from adalflow.core.model_client import ModelClient
from adalflow.core.embedder import Embedder
from adalflow.core.string_parser import (
    YamlParser,
    JsonParser,
    IntParser,
    FloatParser,
    ListParser,
    BooleanParser,
)
from adalflow.components.output_parsers import (
    YamlOutputParser,
    JsonOutputParser,
    ListOutputParser,
)
from adalflow.components.output_parsers.dataclass_parser import DataClassParser
from adalflow.core.prompt_builder import Prompt
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
    AnthropicAPIClient,
    CohereAPIClient,
)

__all__ = [
    "Component",
    "fun_to_component",
    "DataClass",
    "Sequential",
    "GradComponent",
    "ModelClient",
    "Generator",
    "Embedder",
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
    # Output Parsers with dataclass formatting
    "YamlOutputParser",
    "JsonOutputParser",
    "ListOutputParser",
    "DataClassParser",
    # Types
    "GeneratorOutput",
    "EmbedderOutput",
    "RetrieverOutput",
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
    "OllamaClient",
    "TransformersClient",
    "AnthropicAPIClient",
    "CohereAPIClient",
]
