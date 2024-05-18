from typing import Optional, Any, Dict, Union, Sequence
from core.data_classes import ModelType
from core.api_client import APIClient

from core.component import Component
import core.functional as F


# TODO: might be better to type hint the output type and add the output processor separately
class Embedder(Component):
    model_type: ModelType = ModelType.EMBEDDER
    model_client: APIClient
    output_processors: Optional[Component]

    def __init__(
        self,
        *,
        model_client: APIClient,
        model_kwargs: Dict = {},
        output_processors: Optional[Component] = None,
    ) -> None:
        super().__init__()
        if not isinstance(model_kwargs, Dict):
            raise ValueError(
                f"{type(self).__name__} requires a dictionary for model_kwargs, not a string"
            )
        self.model_kwargs = model_kwargs.copy()
        if "model" not in model_kwargs:
            raise ValueError(
                f"{type(self).__name__} requires a 'model' to be passed in the model_kwargs"
            )
        self.model_client = model_client
        self.output_processors = output_processors
        self.model_client._init_sync_client()

    def update_default_model_kwargs(self, **model_kwargs) -> Dict:
        return F.compose_model_kwargs(self.model_kwargs, model_kwargs)

    def _pre_call(self, input: Union[str, Sequence[str]], model_kwargs: Dict) -> Dict:
        composed_model_kwargs = self.update_default_model_kwargs(**model_kwargs)
        return composed_model_kwargs

    def _post_call(self, response: Any) -> Any:
        if self.output_processors:
            return self.output_processors(response)
        return response

    def call(
        self,
        *,
        input: Union[str, Sequence[str]],
        model_kwargs: Optional[Dict] = {},
    ) -> Any:
        composed_model_kwargs = self._pre_call(input, model_kwargs)
        response = self.model_client.call(
            input=input, model_kwargs=composed_model_kwargs, model_type=self.model_type
        )
        return self._post_call(response)

    async def acall(
        self,
        *,
        input: Union[str, Sequence[str]],
        model_kwargs: Optional[Dict] = {},
    ) -> Any:
        composed_model_kwargs = self._pre_call(input, model_kwargs)
        response = await self.model_client.acall(
            input=input, model_kwargs=composed_model_kwargs, model_type=self.model_type
        )
        return self._post_call(response)

    def extra_repr(self) -> str:
        s = f"model_kwargs={self.model_kwargs}"
        return s
