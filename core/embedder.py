from typing import Optional, Any, Dict
from core.data_classes import EmbedderOutput, ModelType

from core.component import Component
import core.functional as F


class Embedder(Component):
    model: ModelType = ModelType.EMBEDDER

    def __init__(self, provider: Optional[str] = None) -> None:
        super().__init__(provider=provider)
        print(f"{type(self).__name__} initialized with model type: {self.model}")

    def compose_model_kwargs(self, **model_kwargs) -> Dict:
        return F.compose_model_kwargs(self.model_kwargs, model_kwargs)

    def call(self, input: Any, **model_kwargs) -> EmbedderOutput:
        raise NotImplementedError(f"{type(self).__name__} must implement call method")
