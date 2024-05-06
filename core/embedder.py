from typing import Optional, Any, Dict
from core.data_classes import EmbedderOutput, ModelType

from core.component import Component
import core.functional as F


class Embedder(Component):
    model_type: ModelType = ModelType.EMBEDDER

    def __init__(self, provider: Optional[str] = None, model_kwargs: Dict = {}) -> None:
        super().__init__()
        self.provider = provider
        self.model_kwargs = model_kwargs.copy()
        if "model" not in model_kwargs:
            raise ValueError(
                f"{type(self).__name__} requires a 'model' to be passed in the model_kwargs"
            )

    def compose_model_kwargs(self, **model_kwargs) -> Dict:
        return F.compose_model_kwargs(self.model_kwargs, model_kwargs)

    def call(self, input: Any, **model_kwargs) -> EmbedderOutput:
        raise NotImplementedError(f"{type(self).__name__} must implement call method")

    def extra_repr(self) -> str:
        s = "provider={provider}, model_kwargs={model_kwargs}"
        return s.format(**self.__dict__)
