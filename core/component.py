from collections import OrderedDict
from typing import Callable, Dict, Any, Optional, List, Tuple, Iterable, Set
import os
from core.data_classes import EmbedderOutput, RetrieverOutput, Chunk
from collections import OrderedDict, abc as container_abcs
import operator
from itertools import islice

# TODO: design hooks.
_global_pre_call_hooks: Dict[int, Callable] = OrderedDict()
__all__ = ["Component", "EmbedderOutput", "OpenAIEmbedder"]


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


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
    # provider: str  # meta data for the developer

    def __init__(self, *args, **kwargs) -> None:
        super().__setattr__("_components", {})
        # super().__init__(*args, **kwargs)
        # super().__setattr__("provider", None)
        # if "provider" in kwargs:
        #     self.provider = kwargs["provider"]

    def __setattr__(self, name: str, value: Any) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        components = self.__dict__.get("_components")
        if isinstance(value, Component):
            if components is None:
                raise AttributeError(
                    "cant assign component before Component.__init__() call"
                )
            remove_from(self.__dict__)
            components[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        if "_components" in self.__dict__:
            components = self.__dict__.get("_components")
            if name in components:
                return components[name]
        # else:
        #     super().__getattr__(name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __delattr__(self, name: str) -> None:
        if name in self._components:
            del self._components[name]
        else:
            super().__delattr__(name)

    def extra_repr(self) -> str:
        return ""

    def _get_name(self):
        return self.__class__.__name__

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, component in self._components.items():
            mod_str = repr(component)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def __call__(self, *args, **kwargs):
        # Default to sync call
        return self.call(*args, **kwargs)

    call: Callable[..., Any] = _call_unimplemented

    async def acall(self, *args, **kwargs):
        pass

    def add_component(self, name: str, component: Optional["Component"]) -> None:
        self._components[name] = component

    def register_subcomponent(
        self, name: str, component: Optional["Component"]
    ) -> None:
        r"""
        Alias for add_component
        """
        self.add_component(name, component)

    def get_subcomponent(self, name: str) -> Optional["Component"]:
        return self._components.get(name)

    def named_children(self) -> Iterable[Tuple[str, "Component"]]:
        r"""
        Returns an iterator over immediate children modules.
        """
        memo = set()
        for name, component in self._components.items():
            if component is not None and component not in memo:
                memo.add(component)
                yield name, component

    def children(self) -> Iterable["Component"]:
        r"""
        Returns an iterator over immediate children modules.
        """
        for name, component in self.named_children():
            yield component

    def named_components(
        self,
        memo: Optional[Set["Component"]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._components.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                yield from module.named_components(
                    memo, submodule_prefix, remove_duplicate
                )

    def components(self) -> Iterable["Component"]:
        r"""
        Returns an iterator over all components in the Module.
        """
        for name, component in self.named_children():
            yield component

    def apply(self: "Component", fn: Callable[["Component", Any], None]) -> None:
        r"""
        Applies a function to all subcomponents.
        # TODO: in what situation we need function apply?
        """
        for name, component in self.children():
            component.apply(fn)
        fn(self)
        return self


from typing import (
    List,
    Dict,
    Any,
    Optional,
    Union,
    Mapping,
    Tuple,
    Iterable,
    Iterator,
    overload,
)


class Sequential(Component):
    r"""A sequential container.

    Components will be added to it in the order they are passed to the constructor.
    """

    _components: Dict[str, Component]  # type: ignore[assignment]

    @overload
    def __init__(self, *args: Component) -> None: ...

    @overload
    def __init__(self, arg: "OrderedDict[str, Component]") -> None: ...

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_component(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_component(str(idx), module)

    def _get_item_by_idx(self, iterator, idx) -> Component:  # type: ignore[misc, type-var]
        """Get the idx-th item of the iterator."""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError(f"index {idx} is out of range")
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx: Union[slice, int]) -> Union["Sequential", Component]:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._components.items())[idx]))
        else:
            return self._get_item_by_idx(self._components.values(), idx)

    def __setitem__(self, idx: int, component: Component) -> None:
        key: str = self._get_item_by_idx(self._components.keys(), idx)
        return setattr(self, key, component)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._components.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._components.keys(), idx)
            delattr(self, key)
        # To preserve numbering
        str_indices = [str(i) for i in range(len(self._components))]
        self._components = OrderedDict(
            list(zip(str_indices, self._components.values()))
        )

    def __len__(self) -> int:
        return len(self._components)

    def append(self, component: Component) -> "Sequential":
        r"""Appends a component to the end of the Sequential."""
        idx = len(self._components)
        self.add_component(str(idx), component)
        return self

    def __add__(self, other) -> "Sequential":
        if not isinstance(other, Sequential):
            ret = Sequential()
            for layer in self:
                ret.append(layer)
            for layer in other:
                ret.append(layer)
            return ret
        else:
            raise ValueError(
                "add operator supports only objects "
                f"of Sequential class, but {str(type(other))} is given."
            )

    def call(self, input: Any) -> Any:
        for component in self._components.values():
            input = component(input)
        return input


class ComponentDict(Component):
    r"""
    We directly used the code in PyTorch's ModuleDict.
    See its design here: /torch/nn/modules/container.py

    Holds submodules in a dictionary.

    :class:`~torch.nn.ModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    :class:`~torch.nn.ModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.ModuleDict.update`, the order of the merged
      ``OrderedDict``, ``dict`` (started from Python 3.6) or another
      :class:`~torch.nn.ModuleDict` (the argument to
      :meth:`~torch.nn.ModuleDict.update`).

    Note that :meth:`~torch.nn.ModuleDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict`` before Python version 3.6) does not
    preserve the order of the merged mapping.

    Args:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.choices = nn.ModuleDict({
                        'conv': nn.Conv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    _components: Dict[str, Component]  # type: ignore[assignment]

    def __init__(self, modules: Optional[Mapping[str, Component]] = None) -> None:
        super().__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key: str) -> Component:
        return self._components[key]

    def __setitem__(self, key: str, component: Component) -> None:
        self.add_component(key, component)

    def __delitem__(self, key: str) -> None:
        del self._components[key]

    def __len__(self) -> int:
        return len(self._components)

    def __iter__(self) -> Iterator[str]:
        return iter(self._components)

    def __contains__(self, key: str) -> bool:
        return key in self._components

    def clear(self) -> None:
        """Remove all items from the ModuleDict."""
        self._components.clear()

    def pop(self, key: str) -> Component:
        r"""Remove key from the ModuleDict and return its module.

        Args:
            key (str): key to pop from the ModuleDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ModuleDict keys."""
        return self._components.keys()

    def items(self) -> Iterable[Tuple[str, Component]]:
        r"""Return an iterable of the ModuleDict key/value pairs."""
        return self._components.items()

    def values(self) -> Iterable[Component]:
        r"""Return an iterable of the ModuleDict values."""
        return self._components.values()

    def update(self, components: Mapping[str, Component]) -> None:
        r"""Update the :class:`~torch.nn.ModuleDict` with key-value pairs from a mapping, overwriting existing keys.

        .. note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~torch.nn.ModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            modules (iterable): a mapping (dictionary) from string to :class:`~torch.nn.Module`,
                or an iterable of key-value pairs of type (string, :class:`~torch.nn.Module`)
        """
        if not isinstance(components, container_abcs.Iterable):
            raise TypeError(
                "ModuleDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(components).__name__
            )

        if isinstance(components, (OrderedDict, ComponentDict, container_abcs.Mapping)):
            for key, module in components.items():
                self[key] = module
        else:
            # modules here can be a list with two items
            for j, m in enumerate(components):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(m).__name__
                    )
                if not len(m) == 2:
                    raise ValueError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " has length " + str(len(m)) + "; 2 is required"
                    )
                # modules can be Mapping (what it's typed at), or a list: [(name1, module1), (name2, module2)]
                # that's too cumbersome to type correctly with overloads, so we add an ignore here
                self[m[0]] = m[1]  # type: ignore[assignment]

    # remove forward alltogether to fallback on Module's _forward_unimplemented


from openai import OpenAI, AsyncOpenAI
from openai import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
    BadRequestError,
)
import backoff
from typing import List, Union, overload
from core.data_classes import EmbedderOutput, ModelType

from core.component import Component
import core.functional as F


class Embedder(Component):
    type: ModelType = ModelType.EMBEDDER
    provider: str

    def __init__(
        self, *, provider: Optional[str] = None, model_kwargs: Optional[Dict] = {}
    ) -> None:
        super().__init__()
        self.provider = provider
        self.model_kwargs = model_kwargs

    def compose_model_kwargs(self, **model_kwargs) -> Dict:
        return F.compose_model_kwargs(self.model_kwargs, model_kwargs)

    def call(self, input: Any, **model_kwargs) -> EmbedderOutput:
        raise NotImplementedError


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


# class GeneratorRunner:
#     """
#     A base class for running a generator.
#     TODO: history
#     """

#     name = "GeneratorRunner"

#     def __init__(
#         self,
#         generator: Model,
#         prompt: str = None,
#         examples: List[str] = [],
#     ):
#         self.generator = generator
#         self.prompt = prompt
#         self.examples = examples
#         self.prompt_template = Template(self.prompt) if prompt else None

#     def __call__(self, **kwargs) -> Any:
#         self.kwargs = kwargs
#         if "examples" in self.kwargs:
#             examples = self.kwargs.get("examples")
#         else:
#             examples = self.examples
#         system_prompt = (
#             self.prompt_template.render(
#                 user_query=self.kwargs.get("input"),
#                 examples=examples,
#             )
#             if self.prompt_template
#             else self.kwargs.get("input")
#         )
#         messages = [{"role": "system", "content": system_prompt}]
#         print(f"messages: {messages}")
#         response = self.generator(messages)
#         return response
