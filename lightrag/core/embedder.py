r"""The component that orchestrates model client (Embedding models in particular) and output processors."""

from typing import Optional, Any, Dict
import logging

from lightrag.core.types import ModelType
from lightrag.core.model_client import ModelClient, API_INPUT_TYPE
from lightrag.core.types import EmbedderOutput
from lightrag.core.component import Component
import lightrag.core.functional as F

EmbedderInputType = API_INPUT_TYPE
EmbedderOutputType = EmbedderOutput

log = logging.getLogger(__name__)


class Embedder(Component):
    r"""
    A user-facing component that orchestrates an embedder model via the model client and output processors.

    Args:
        model_client (ModelClient): The model client to use for the embedder.
        model_kwargs (Dict[str, Any], optional): The model kwargs to pass to the model client. Defaults to {}.
        output_processors (Optional[Component], optional): The output processors after model call. Defaults to None.
            If you want to add further processing, it should operate on the ``EmbedderOutput`` data type.

    Note:
        The ``output_processors`` will be applied only on the data field of ``EmbedderOutput``, which is a list of ``Embedding``.
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
        # if "model" not in model_kwargs:
        #     raise ValueError(
        #         f"{type(self).__name__} requires a 'model' to be passed in the model_kwargs"
        #     )
        self.model_client = model_client
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
        log.debug(f"api_kwargs: {api_kwargs}")
        return api_kwargs

    def _post_call(self, response: Any) -> EmbedderOutputType:
        embedding_output: EmbedderOutputType = (
            self.model_client.parse_embedding_response(response)
        )
        data = embedding_output.data
        if not embedding_output.error and data is not None:
            if self.output_processors:
                embedding_output.data = self.output_processors(data)

        return embedding_output

    def call(
        self,
        *,
        input: EmbedderInputType,
        model_kwargs: Optional[Dict] = {},
    ) -> EmbedderOutputType:
        log.debug(f"Calling {self.__class__.__name__} with input: {input}")
        api_kwargs = self._pre_call(input=input, model_kwargs=model_kwargs)
        response = self.model_client.call(
            api_kwargs=api_kwargs, model_type=self.model_type
        )

        output = self._post_call(response)
        log.debug(f"Output from {self.__class__.__name__}: {output}")
        return output

    async def acall(
        self,
        *,
        input: EmbedderInputType,
        model_kwargs: Optional[Dict] = {},
    ) -> EmbedderOutputType:
        log.debug(f"Calling {self.__class__.__name__} with input: {input}")
        composed_model_kwargs = self._pre_call(input=input, model_kwargs=model_kwargs)
        response = await self.model_client.acall(
            input=input, model_kwargs=composed_model_kwargs, model_type=self.model_type
        )
        output = self._post_call(response)
        log.debug(f"Output from {self.__class__.__name__}: {output}")
        return output

    def extra_repr(self) -> str:
        s = f"model_kwargs={self.model_kwargs}"
        return s
