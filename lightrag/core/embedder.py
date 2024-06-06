from typing import Optional, Any, Dict

from lightrag.core.types import ModelType
from lightrag.core.model_client import ModelClient, API_INPUT_TYPE
from lightrag.core.component import Component
import lightrag.core.functional as F

EmbedderInputType = API_INPUT_TYPE
EmbedderOutputType = Any


class Embedder(Component):
    r"""
    A user-facing component that orchestrates an embedder model via the model client and output processors.
    """

    model_type: ModelType = ModelType.EMBEDDER
    model_client: ModelClient
    output_processors: Optional[Component]

    def __init__(
        self,
        *,
        model_client: ModelClient,
        model_kwargs: Dict[str, Any] = {},
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
        self.model_client = model_client()
        self.output_processors = output_processors

    def update_default_model_kwargs(self, **model_kwargs) -> Dict:
        return F.compose_model_kwargs(self.model_kwargs, model_kwargs)

    def _pre_call(self, input: EmbedderInputType, model_kwargs: Dict) -> Dict:
        # step 1: combine the model_kwargs with the default model_kwargs
        composed_model_kwargs = self.update_default_model_kwargs(**model_kwargs)
        # step 2: convert the input to the api_kwargs
        api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
            input=input,
            model_kwargs=composed_model_kwargs,
            model_type=self.model_type,
        )
        print(f"api_kwargs {api_kwargs}")
        return api_kwargs

    def _post_call(self, response: Any) -> EmbedderOutputType:
        if self.output_processors:
            return self.output_processors(response)
        return response

    def call(
        self,
        *,
        input: EmbedderInputType,
        model_kwargs: Optional[Dict] = {},
    ) -> EmbedderOutputType:
        print(f"start to embed the data")
        api_kwargs = self._pre_call(input=input, model_kwargs=model_kwargs)
        print(f"start to call")
        response = self.model_client.call(
            api_kwargs=api_kwargs, model_type=self.model_type
        )
        return self._post_call(response)

    async def acall(
        self,
        *,
        input: EmbedderInputType,
        model_kwargs: Optional[Dict] = {},
    ) -> EmbedderOutputType:
        composed_model_kwargs = self._pre_call(model_kwargs)
        response = await self.model_client.acall(
            input=input, model_kwargs=composed_model_kwargs, model_type=self.model_type
        )
        return self._post_call(response)

    def extra_repr(self) -> str:
        s = f"model_kwargs={self.model_kwargs}"
        return s
