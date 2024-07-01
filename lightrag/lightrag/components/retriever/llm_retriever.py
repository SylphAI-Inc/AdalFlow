"""LLM as retriever module."""

from typing import Optional, Any, Dict, Callable
import logging

from lightrag.core.retriever import (
    Retriever,
)
from lightrag.core.generator import Generator
from lightrag.core.model_client import ModelClient
from lightrag.core.string_parser import ListParser
from lightrag.core.types import (
    GeneratorOutput,
    RetrieverOutput,
    RetrieverDocumentsType,
    RetrieverStrQueryType,
    RetrieverStrQueriesType,
    RetrieverOutputType,
)

log = logging.getLogger(__name__)


DEFAULT_LLM_AS_RETRIEVER_PROMPT_TEMPLATE = r"""<SYS>
You are a retriever. Given a list of documents, you will retrieve the top_k {{top_k}} most relevant documents and output the indices (int) as a list:
[<index of the most relevant top_k options>]
<Documents>
{% for doc in documents %}
```Index {{ loop.index - 1 }}. {{ doc }}```
______________
{% endfor %}
</Documents>
</SYS>
Query: {{ input_str }}
You:
"""


class LLMRetriever(Retriever[str, RetrieverStrQueryType]):
    __doc__ = r"""Use LLM to access the query and the documents to retrieve the top k relevant indices of the documents.

    Users can follow this example and to customize the prompt or additionally ask it to output score along with the indices.

    Args:
        top_k (Optional[int], optional): top k documents to fetch. Defaults to 1.
        model_client (ModelClient): the model client to use.
        model_kwargs (Dict[str, Any], optional): the model kwargs. Defaults to {}.

    .. note::
        There is chance some queries might fail, which will lead to empty response None for that query in the List of RetrieverOutput. Users should handle this case.
    """

    def __init__(
        self,
        *,
        top_k: Optional[int] = 1,
        # the genearator kwargs
        model_client: ModelClient,
        model_kwargs: Dict[str, Any] = {},
        documents: Optional[RetrieverDocumentsType] = None,
        document_map_func: Optional[Callable[[Any], str]] = None,
    ):
        super().__init__()
        self.reset_index()
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=DEFAULT_LLM_AS_RETRIEVER_PROMPT_TEMPLATE,
            preset_prompt_kwargs={"top_k": top_k},
            output_processors=ListParser(),
        )

        self.top_k = top_k
        self.model_kwargs = model_kwargs

        if documents:
            self.build_index_from_documents(documents, document_map_func)

    def reset_index(self):
        self.indexed = False
        self.total_documents = 0

    def build_index_from_documents(
        self,
        documents: RetrieverDocumentsType,
        document_map_func: Optional[Callable[[Any], str]] = None,
    ):
        r"""prepare the user query input for the retriever"""
        if document_map_func:
            documents = [document_map_func(doc) for doc in documents]
        else:
            documents = documents
        self.total_documents = len(documents)
        self.generator.prompt.update_preset_prompt_kwargs(documents=documents)
        self.indexed = True

    def call(
        self,
        input: RetrieverStrQueriesType,
        top_k: Optional[int] = None,
        model_kwargs: Dict[str, Any] = {},
    ) -> RetrieverOutputType:
        """Retrieve the k relevant documents.

        Args:
            query_or_queries (RetrieverStrQueriesType): a string or a list of strings.
            top_k (Optional[int], optional): top k documents to fetch. Defaults to None.
            model_kwargs (Dict[str, Any], optional): the model kwargs.
             You can switch to another model provided by the same model client without reinitializing the retriever. Defaults to {}.

        Returns:
            RetrieverOutputType: the developers should be aware that the returned ``LLMRetrieverOutputType`` is actually a list of GeneratorOutput(:class:`GeneratorOutput <lightrag.core.types.GeneratorOutput>`), post processing is required depends on how you instruct the model to output in the prompt and what ``output_processors`` you set up.
            E.g. If the prompt is to output a list of indices and the ``output_processors`` is ``ListParser()``, then it return: GeneratorOutput(data=[indices], error=None, raw_response='[indices]')
        """
        assert self.indexed, "The retriever is not indexed yet."
        top_k = top_k or self.top_k
        queries = input if isinstance(input, list) else [input]
        retrieved_outputs: RetrieverOutputType = []

        for query in queries:
            prompt_kwargs = {
                "input_str": query,
                "top_k": top_k,
            }
            model_kwargs_to_use = self.model_kwargs.copy()
            model_kwargs_to_use.update(model_kwargs)
            response: GeneratorOutput = self.generator(
                prompt_kwargs=prompt_kwargs, model_kwargs=model_kwargs_to_use
            )
            if response.error or response.data is None:
                log.error(f"query: {query} failed to retrieve")
                log.error(f"error_message: {response.error}")
                log.error(f"raw_response: {response.raw_response}")
                log.error(f"response: {response.data}")
                retrieved_outputs.append(RetrieverOutput(doc_indices=[]))
                continue
            retrieved_outputs.append(
                RetrieverOutput(
                    doc_indices=response.data,
                    query=query,
                )
            )
        return retrieved_outputs

    def _extra_repr(self) -> str:
        s = f"top_k={self.top_k}, total_documents={self.total_documents},"
        return s
