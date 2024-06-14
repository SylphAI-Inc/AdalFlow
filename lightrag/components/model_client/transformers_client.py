"""Huggingface transformers ModelClient integration."""

from typing import Any, Dict, Union, List, Optional, Tuple
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
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForSequenceClassification,
    )

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


class TransformerReranker:
    __doc__ = r"""Local model SDK for a reranker model using transformers.

    References:
    - model: https://huggingface.co/BAAI/bge-reranker-base
    - paper: https://arxiv.org/abs/2309.07597

    note:
    If you are using Macbook M1 series chips, you need to ensure ``torch.device("mps")`` is set.
    """
    models: Dict[str, type] = {}

    def __init__(self, model_name: Optional[str] = "BAAI/bge-reranker-base"):
        self.model_name = model_name or "BAAI/bge-reranker-base"
        if model_name is not None:
            self.init_model(model_name=model_name)

    @lru_cache(None)
    def init_model(self, model_name: str):
        try:
            print(f"Loading model {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"get tokenizer: {self.tokenizer}")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            # Check device availability and set the device
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("Using CUDA (GPU) for inference.")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                print("Using MPS (Apple Silicon) for inference.")
            else:
                device = torch.device("cpu")
                print("Using CPU for inference.")

            # Move model to the selected device
            self.device = device
            self.model.to(device)
            print(f"model: {self.model}")
            self.model.eval()
            print(f"model: {self.model}")
            # register the model
            self.models[model_name] = self.model  # TODO: better model registration
            log.info(f"Done loading model {model_name}")

        except Exception as e:
            log.error(f"Error loading model {model_name}: {e}")
            raise e

    def infer_bge_reranker_base(
        self,
        input=List[Tuple[str, str]],  # list of pairs of the query and the candidate
    ) -> List[float]:
        model = self.models.get(self.model_name, None)
        if model is None:
            # initialize the model
            self.init_model(self.model_name)

        with torch.no_grad():
            inputs = self.tokenizer(
                input,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            print(inputs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            print(f"model: {model}")
            scores = (
                model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )
            print(scores)
            # apply sigmoid to get the scores
            scores = F.sigmoid(scores)
        return scores.tolist()

    def __call__(self, **kwargs):
        r"""Ensure "model" and "input" are in the kwargs."""
        if "model" not in kwargs:
            raise ValueError("model is required")

        # if "mock" in kwargs and kwargs["mock"]:
        #     import numpy as np

        #     scores = np.array([np.random.rand(1).tolist()])
        #     return scores
        # load files and models, cache it for the next inference
        model_name = kwargs["model"]
        # inference the model
        if model_name == self.model_name:
            return self.infer_bge_reranker_base(kwargs["input"])
        else:
            raise ValueError(f"model {model_name} is not supported")


class TransformersClient(ModelClient):
    __doc__ = r"""LightRAG API client for transformers.

    Use: ``ls ~/.cache/huggingface/hub `` to see the cached models.
    """

    support_models = {
        "thenlper/gte-base": {
            "type": ModelType.EMBEDDER,
        },
        "BAAI/bge-reranker-base": {
            "type": ModelType.RERANKER,
        },
    }

    def __init__(self, model_name: Optional[str] = None) -> None:
        super().__init__()
        self._model_name = model_name or "thenlper/gte-base"
        assert (
            self._model_name in self.support_models
        ), f"model {self._model_name} is not supported"
        if self._model_name == "thenlper/gte-base":
            self.sync_client = self.init_sync_client()
        elif self._model_name == "BAAI/bge-reranker-base":
            self.reranker_client = None
        self.async_client = None

    def init_sync_client(self):
        return TransformerEmbedder()

    def init_reranker_client(self):
        return TransformerReranker()

    def parse_embedding_response(self, response: Any) -> EmbedderOutput:
        embeddings: List[Embedding] = []
        for idx, emb in enumerate(response):
            embeddings.append(Embedding(index=idx, embedding=emb))
        response = EmbedderOutput(data=embeddings)
        return response

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        assert "model" in api_kwargs, "model is required"
        assert (
            api_kwargs["model"] in self.support_models
        ), f"model {api_kwargs['model']} is not supported"
        if (
            model_type == ModelType.EMBEDDER
            and "model" in api_kwargs
            and api_kwargs["model"] == "thenlper/gte-base"
        ):
            if self.sync_client is None:
                self.sync_client = self.init_sync_client()
            return self.sync_client(**api_kwargs)
        elif (  # reranker
            model_type == ModelType.RERANKER
            and "model" in api_kwargs
            and api_kwargs["model"] == "BAAI/bge-reranker-base"
        ):
            print(f"reranker_client: {self.reranker_client}")
            if not hasattr(self, "reranker_client") or self.reranker_client is None:
                print("init reranker client")
                self.reranker_client = self.init_reranker_client()
                print(f"reranker_client: {self.reranker_client}")
            return self.reranker_client(**api_kwargs)

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
        elif model_type == ModelType.RERANKER:
            final_model_kwargs["input"] = input
            return final_model_kwargs
        else:
            raise ValueError(f"model_type {model_type} is not supported")
