"""Huggingface transformers ModelClient integration."""

from typing import Any, Dict, Union, List, Optional
import logging
from functools import lru_cache
from lightrag.core.types import EmbedderOutput
import torch.nn.functional as F

try:
    import torch
except ImportError:
    raise ImportError("Please install torch with: pip install torch")
from torch import Tensor

try:
    import transformers
    from transformers import AutoTokenizer, AutoModel

except ImportError:
    raise ImportError("Please install transformers with: pip install transformers")

from lightrag.core.model_client import ModelClient
from lightrag.core.types import ModelType, Embedding

from lightrag.core.component import Component

log = logging.getLogger(__name__)


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# TODO: provide a standard api for embedding and chat models used in local model SDKs
class TransformerEmbedder:
    """Local model SDK for transformers.


    There are two ways to run transformers:
    (1) model and then run model inference
    (2) Pipeline and then run pipeline inference

    This file demonstrates how to
    (1) create a torch model inference component:  TransformerEmbedder which equalize to OpenAI(), the SyncAPIClient
    (2) Convert this model inference component to LightRAG API client: TransformersClient

    The is now just an exmplary component that initialize a certain model from transformers and run inference on it.
    It is not tested on all transformer models yet. It might be necessary to write one for each model.

    References:
    - transformers: https://huggingface.co/docs/transformers/en/index
    - thenlper/gte-base model:https://huggingface.co/thenlper/gte-base
    """

    models: Dict[str, type] = {}

    def __init__(self, model_name: Optional[str] = "thenlper/gte-base"):
        super().__init__()

        if model_name is not None:
            self.init_model(model_name=model_name)

    @lru_cache(None)
    def init_model(self, model_name: str):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            # register the model
            self.models[model_name] = self.model
            log.info(f"Done loading model {model_name}")

        except Exception as e:
            log.error(f"Error loading model {model_name}: {e}")
            raise e

    def infer_gte_base_embedding(
        self,
        input=Union[str, List[str]],
        tolist: bool = True,
    ):
        model = self.models.get("thenlper/gte-base", None)
        if model is None:
            # initialize the model
            self.init_model("thenlper/gte-base")

        if isinstance(input, str):
            input = [input]
        # Tokenize the input texts
        batch_dict = self.tokenizer(
            input, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        outputs = model(**batch_dict)
        embeddings = average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        # (Optionally) normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        if tolist:
            embeddings = embeddings.tolist()
        return embeddings

    def __call__(self, **kwargs):
        if "model" not in kwargs:
            raise ValueError("model is required")

        if "mock" in kwargs and kwargs["mock"]:
            import numpy as np

            embeddings = np.array([np.random.rand(768).tolist()])
            return embeddings
        # load files and models, cache it for the next inference
        model_name = kwargs["model"]
        # inference the model
        if model_name == "thenlper/gte-base":
            return self.infer_gte_base_embedding(kwargs["input"])
        else:
            raise ValueError(f"model {model_name} is not supported")


class TransformersClient(ModelClient):
    def __init__(self) -> None:
        super().__init__()
        self.sync_client = self.init_sync_client()
        self.async_client = None
        support_model_list = {
            "thenlper/gte-base": {
                "type": ModelType.EMBEDDER,
            }
        }

    def init_sync_client(self):
        return TransformerEmbedder()

    def parse_embedding_response(self, response: Any) -> EmbedderOutput:
        # convert list of float to list of embedding
        # TODO: need to simplify this, Embedding data type does not seem necessary
        embeddings: List[Embedding] = []
        for idx, emb in enumerate(response):
            embeddings.append(Embedding(index=idx, embedding=emb))
        response = EmbedderOutput(data=embeddings)
        return response

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        return self.sync_client(**api_kwargs)

    def convert_inputs_to_api_kwargs(
        self,
        input: Any,
        model_kwargs: dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> dict:
        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            final_model_kwargs["input"] = input
            return final_model_kwargs
        else:
            raise ValueError(f"model_type {model_type} is not supported")


if __name__ == "__main__":

    def test_transformer_embedder():
        transformer_embedder_model = "thenlper/gte-base"
        transformer_embedder_model_component = TransformerEmbedder()
        print(
            f"Testing transformer embedder with model {transformer_embedder_model_component}"
        )
        print("Testing transformer embedder")
        output = transformer_embedder_model_component(
            model=transformer_embedder_model, input="Hello world"
        )
        print(output)

    def test_transformer_client():
        transformer_client = TransformersClient()
        print("Testing transformer client")
        # run the model
        kwargs = {
            "model": "thenlper/gte-base",
            "mock": False,
        }
        api_kwargs = transformer_client.convert_inputs_to_api_kwargs(
            input="Hello world",
            model_kwargs=kwargs,
            model_type=ModelType.EMBEDDER,
        )
        print(api_kwargs)
        output = transformer_client.call(
            api_kwargs=api_kwargs, model_type=ModelType.EMBEDDER
        )

        print(transformer_client)
        print(output)

    # test transfomer embedding
    from transformers import file_utils

    # print(file_utils.default_cache_path)
    # from transformers import TRANSFORMERS_CACHE

    # print(TRANSFORMERS_CACHE)

    # import shutil

    # shutil.rmtree(TRANSFORMERS_CACHE)
    test_transformer_embedder()
    # test_transformer_client()
