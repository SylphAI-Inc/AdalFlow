r"""We let users install the required SDKs conditionally for our integrated model providers."""

# try:
#     from .openai_client import *
# except ImportError:

#     pass
# try:
#     from .groq_client import *
# except ImportError:
#     pass
# try:
#     from .anthropic_client import *
# except ImportError:
#     pass
# try:
#     from .transformers_client import *
# except ImportError:
#     pass
# try:
#     from .google_client import *
# except ImportError:
#     pass

from .anthropic_client import AnthropicAPIClient
from .google_client import GoogleGenAIClient
from .groq_client import GroqAPIClient
from .openai_client import OpenAIClient
from .transformers_client import (
    TransformersClient,
    TransformerEmbedder,
    TransformerReranker,
)
from lightrag.utils.registry import EntityMapping


__all__ = [
    "AnthropicAPIClient",
    "GoogleGenAIClient",
    "GroqAPIClient",
    "OpenAIClient",
    "TransformersClient",
    "TransformerEmbedder",
    "TransformerReranker",
]
for name in __all__:
    EntityMapping.register(name, globals()[name])
