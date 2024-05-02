from collections import OrderedDict
from typing import Callable, Dict, Any, Optional
import os

# TODO: design hooks.
_global_pre_call_hooks: Dict[int, Callable] = OrderedDict()
__all__ = ["Component", "EmbedderOutput", "OpenAIEmbedder"]


def _call_unimplemented(self, *input: Any) -> None:
    r"""
    Define the call method for the component.
    Should be overriden by all subclasses.
    """
    raise NotImplementedError(
        f'Component {type(self).__name__} is missing the required "call" method.'
    )


class Component:
    r"""
    Component defines all functional base classes such as Embedder, Retriever, Generator.

    We purposly avoid using the name "Module" to avoid confusion with PyTorch's nn.Module.
    As we consider 'Component' to be an extension to 'Moduble' as if you use a local llm model
    for the Generator, you might need the 'Module' within the 'Component'.

    But 'Component' follows a similar design pattern to 'Module' in PyTorch.

    (1) 'Module' does not have async function because of GPU's inherent parallelism.
     But we need to support async functions.
     call and acall should call functions.
    (2) All components can be running local or APIs. 'Component' can deal with API calls, so we need support retries and rate limits.
    """

    _version: int = 0
    # TODO: the type of module, is it OrderedDict or just Dict?
    _components: Dict[str, Optional["Component"]]
    provider: str  # meta data for the developer

    def __init__(self, *args, **kwargs) -> None:
        super().__setattr__("_components", {})
        super().__setattr__("provider", None)
        if "provider" in kwargs:
            self.provider = kwargs["provider"]

    def __call__(self, *args, **kwargs):
        # Default to sync call
        return self.call(*args, **kwargs)

    call: Callable[..., Any] = _call_unimplemented

    async def acall(self, *args, **kwargs):
        pass

    def register_subcomponent(
        self, name: str, component: Optional["Component"]
    ) -> None:
        self._components[name] = component

    def get_subcomponent(self, name: str) -> Optional["Component"]:
        return self._components.get(name)


from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
    BadRequestError,
)
import backoff
from typing import List, Union, overload


class EmbedderOutput:
    embeddings: List[List[float]]  # batch_size X embedding_size
    usage: Dict[str, Any]  # api or model usage

    def __init__(self, embeddings: List[List[float]], usage: Dict[str, Any]):
        self.embeddings = embeddings
        self.usage = usage

    def __repr__(self) -> str:
        return f"EmbedderOutput(embeddings={self.embeddings[0:5]}, usage={self.usage})"

    def __str__(self):
        return self.__repr__()


class Model(Component):
    r"""
    Base class for most Model inference (or potentially training). If your Model does not fit this pattern, you can extend Component directly.
    Either local or via API calls.
    Support Embedder, LLM Generator.
    """

    # TODO: allow for specifying the model type, e.g. "embedder", "LLM",
    # TODO: support image models
    type: Optional[str]

    def __init__(
        self, provider: str, type: Optional[str] = None, **model_kwargs
    ) -> None:
        super().__init__(provider=provider)
        if "model" not in model_kwargs:
            raise ValueError(
                f"{type(self).__name__} requires a 'model' to be passed in the model_kwargs"
            )
        self.type = type
        self.model_kwargs = model_kwargs

    # define two types, one for embeddings, and one for generator with completions
    @overload
    def call(self, input: str, **model_kwargs) -> EmbedderOutput: ...

    @overload
    def call(self, input: List[str], **model_kwargs) -> EmbedderOutput: ...

    @overload
    def call(self, input: List[Dict], **model_kwargs) -> Any: ...

    def call(self, input: Any, **model_kwargs) -> Any:
        raise NotImplementedError(
            f"Model {type(self).__name__} is missing the required 'call' method."
        )

    def combine_kwargs(self, **model_kwargs) -> Dict:
        r"""
        Combine the default model, model_kwargs with the passed model_kwargs.
        Example:
        model_kwargs = {"temperature": 0.5, "model": "gpt-3.5-turbo"}
        self.model_kwargs = {"model": "gpt-3.5"}
        combine_kwargs(model_kwargs) => {"temperature": 0.5, "model": "gpt-3.5-turbo"}

        """
        pass_model_kwargs = self.model_kwargs.copy()

        if pass_model_kwargs:
            model_kwargs.update(model_kwargs)
        return pass_model_kwargs

    def parse_completion(self, completion: ChatCompletion) -> str:
        """
        Parse the completion to a structure your sytem standarizes. (here is str)
        """
        return completion.choices[0].message.content


class OpenAIEmbedder(Model):
    def __init__(self, provider: str = "OpenAI", **model_kwargs) -> None:
        type = "embedder"
        super().__init__(provider, type, **model_kwargs)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Environment variable OPENAI_API_KEY must be set")
        self.client = OpenAI()

    @staticmethod
    def _process_text(text: str) -> str:
        """
        This is specific to OpenAI API, as removing new lines could have better performance
        """
        text = text.replace("\n", " ")
        return text

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
        ),
        max_time=5,
    )
    def call(
        self,
        input: Any,
        **model_kwargs,  # overwrites the default kwargs
    ) -> EmbedderOutput:
        """
        Automatically handles retries for the above exceptions
        TODO: support async calls
        """
        formulated_inputs = []
        if isinstance(input, str):
            formulated_inputs.append(self._process_text(input))
        else:
            for query in input:
                formulated_inputs.append(self._process_text(query))

        num_queries = len(formulated_inputs)

        # check overrides for kwargs
        pass_model_kwargs = self.combine_kwargs(**model_kwargs)

        print(f"kwargs: {model_kwargs}")

        response = self.client.embeddings.create(
            input=formulated_inputs, **pass_model_kwargs
        )
        usage = response.usage
        embeddings = [data.embedding for data in response.data]
        assert (
            len(embeddings) == num_queries
        ), f"Number of embeddings {len(embeddings)} is not equal to the number of queries {num_queries}"
        return EmbedderOutput(embeddings=embeddings, usage=usage)


class OpenAIGenerator(Model):
    name = "OpenAIGenerator"

    def __init__(self, provider: str, **kwargs) -> None:
        type = "generator"
        super().__init__(provider, type, **kwargs)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Environment variable OPENAI_API_KEY must be set")
        self.sync_client = OpenAI()
        self.async_client = None  # only initialize when needed

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
            BadRequestError,
        ),
        max_time=5,
    )
    def call(
        self, input: List[Dict], model: Optional[str] = None, **model_kwargs
    ) -> str:
        """
        input are messages in the format of [{"role": "user", "content": "Hello"}]
        """
        if model:  # overwrite the default model
            self.model = model
        combined_kwargs = self.combine_kwargs(**model_kwargs)
        if not self.sync_client:
            self.sync_client = OpenAI()
        completion = self.sync_client.chat.completions.create(
            messages=input, **combined_kwargs
        )
        # print(f"completion: {completion}")
        response = self.parse_completion(completion)
        return response

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
        ),
        max_time=5,
    )
    async def acall(
        self, messages: List[Dict], model: Optional[str] = None, **kwargs
    ) -> str:
        if model:
            self.model = model

        combined_kwargs = self.combine_kwargs(**kwargs)
        if not self.async_client:
            self.async_client = AsyncOpenAI()
        completion = await self.async_client.chat.completions.create(
            messages=messages, **combined_kwargs
        )
        response = self.parse_completion(completion)
        return response


from core.light_rag import Chunk


class RetrieverOutput:
    """
    Retrieved result per query
    """

    chunks: List[Chunk]
    query: Optional[str] = None

    def __init__(self, chunks: List[Chunk], query: Optional[str] = None):
        self.chunks = chunks
        self.query = query

    def __repr__(self) -> str:
        return f"RetrieverOutput(chunks={self.chunks[0:5]}, query={self.query})"

    def __str__(self):
        return self.__repr__()


import numpy as np
from copy import deepcopy
from typing import List, Optional, Union
import faiss


class FAISSRetriever(Component):
    """
    https://github.com/facebookresearch/faiss
    The retriever uses in-memory Faiss index to retrieve the top k chunks
    d: dimension of the vectors
    xb: number of vectors to put in the index
    xq: number of queries
    The data type dtype must be float32.
    Note: When the num of chunks are less than top_k, the last columns will be -1

    Other index options:
    - faiss.IndexFlatL2: L2 or Euclidean distance, [-inf, inf]
    - faiss.IndexFlatIP: inner product of normalized vectors will be cosine similarity, [-1, 1]

    We choose cosine similarity and convert it to range [0, 1] by adding 1 and dividing by 2 to simulate probability
    """

    name = "FAISSRetriever"

    def __init__(
        self,
        top_k: int = 3,
        d: int = 768,
        chunks: Optional[List[Chunk]] = None,
        vectorizer: Optional[Component] = None,
    ):
        super().__init__(provider="Meta")
        self.d = d
        self.index = faiss.IndexFlatIP(
            d
        )  # inner product of normalized vectors will be cosine similarity, [-1, 1]

        self.vectorizer = vectorizer  # used to vectorize the queries
        if chunks:
            self.set_chunks(chunks)
        self.top_k = top_k

    def reset(self):
        self.index.reset()
        self.chunks: List[Chunk] = []
        self.total_chunks: int = 0

    def set_chunks(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.total_chunks = len(chunks)
        embeddings = [chunk.vector for chunk in chunks]
        xb = np.array(embeddings, dtype=np.float32)
        self.index.add(xb)

    def _convert_cosine_similarity_to_probability(self, D: np.ndarray) -> np.ndarray:
        D = (D + 1) / 2
        D = np.round(D, 3)
        return D

    def _to_retriever_output(
        self, Ind: np.ndarray, D: np.ndarray
    ) -> List[RetrieverOutput]:
        output: List[RetrieverOutput] = []
        # Step 1: Filter out the -1, -1 columns along with its scores when top_k > len(chunks)
        if -1 in Ind:
            valid_columns = ~np.any(Ind == -1, axis=0)

            D = D[:, valid_columns]
            Ind = Ind[:, valid_columns]
        # Step 2: processing rows (one query at a time)
        for row in zip(Ind, D):
            indexes, distances = row
            chunks: List[Chunk] = []
            for index, distance in zip(indexes, distances):
                chunk: Chunk = deepcopy(self.chunks[index])
                chunk.score = distance
                chunks.append(chunk)

            output.append(RetrieverOutput(chunks=chunks))

        return output

    def __call__(
        self, query_or_queries: Union[str, List[str]], top_k: Optional[int] = None
    ) -> List[RetrieverOutput]:
        # if you pass a single query, you should access the first element of the list
        if self.index.ntotal == 0:
            raise ValueError(
                "Index is empty. Please set the chunks to build the index from"
            )
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [q for q in queries if q]  # Filter empty queries
        queries_embeddings = self.vectorizer(queries).embeddings
        xq = np.array(queries_embeddings, dtype=np.float32)
        D, Ind = self.index.search(xq, top_k if top_k else self.top_k)
        D = self._convert_cosine_similarity_to_probability(D)
        retrieved_output = self._to_retriever_output(Ind, D)
        for i, output in enumerate(retrieved_output):
            output.query = queries[i]
        return retrieved_output


from jinja2 import Template


class GeneratorRunner:
    """
    A base class for running a generator.
    TODO: history
    """

    name = "GeneratorRunner"

    def __init__(
        self,
        generator: Model,
        prompt: str = None,
        examples: List[str] = [],
    ):
        self.generator = generator
        self.prompt = prompt
        self.examples = examples
        self.prompt_template = Template(self.prompt) if prompt else None

    def __call__(self, **kwargs) -> Any:
        self.kwargs = kwargs
        if "examples" in self.kwargs:
            examples = self.kwargs.get("examples")
        else:
            examples = self.examples
        system_prompt = (
            self.prompt_template.render(
                user_query=self.kwargs.get("input"),
                examples=examples,
            )
            if self.prompt_template
            else self.kwargs.get("input")
        )
        messages = [{"role": "system", "content": system_prompt}]
        print(f"messages: {messages}")
        response = self.generator(messages)
        return response
