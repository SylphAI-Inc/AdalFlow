from typing import List, Optional, Any, Dict
import logging

from lightrag.core.retriever import Retriever, RetrieverInputType, RetrieverOutputType
from lightrag.core.generator import Generator
from lightrag.core.model_client import ModelClient
from lightrag.core.string_parser import ListParser
from lightrag.core.types import GeneratorOutput

log = logging.getLogger(__name__)


DEFAULT_LLM_AS_RETRIEVER_PROMPT_TEMPLATE = r"""<SYS>
Your are a retriever. Given a list of documents in the context, \
you will retrieve a list of {{top_k}} indices(int) of the documents that are most relevant to the query. You will output a list as follows:
[<id from the most relevent with top_k options>]
<Documents>
{% for doc in documents %}
```{{ loop.index - 1}}. {{doc}}```
{% endfor %}
</Documents>
</SYS>
Query: {{input_str}}
You:
"""


class LLMRetriever(Retriever):
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
    ):
        super().__init__()
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=DEFAULT_LLM_AS_RETRIEVER_PROMPT_TEMPLATE,
            preset_prompt_kwargs={"top_k": top_k},
            output_processors=ListParser(),
        )

        self.top_k = top_k

    def build_index_from_documents(
        self,
        documents: List[str],
    ):
        r"""prepare the user query input for the retriever"""
        documents_to_use = documents.copy()
        self.generator.prompt.update_preset_prompt_kwargs(documents=documents_to_use)

    def retrieve(
        self, query_or_queries: RetrieverInputType, top_k: Optional[int] = None
    ) -> RetrieverOutputType:
        """Retrieve the k relevant documents.

        Args:
            query_or_queries (RetrieverInputType): a string or a list of strings.
            top_k (Optional[int], optional): top k documents to fetch. Defaults to None.

        Returns:
            RetrieverOutputType: the developers should be aware that the returned ``LLMRetrieverOutputType`` is actually a list of GeneratorOutput(:class:`GeneratorOutput <lightrag.core.types.GeneratorOutput>`), post processing is required depends on how you instruct the model to output in the prompt and what ``output_processors`` you set up.
            E.g. If the prompt is to output a list of indices and the ``output_processors`` is ``ListParser()``, then it return: GeneratorOutput(data=[indices], error=None, raw_response='[indices]')
        """
        top_k = top_k or self.top_k
        queries = (
            query_or_queries
            if isinstance(query_or_queries, list)
            else [query_or_queries]
        )
        retrieved_outputs: RetrieverOutputType = []

        for query in queries:
            prompt_kwargs = {
                "input_str": query,
                "top_k": top_k,
            }
            response: GeneratorOutput = self.generator(prompt_kwargs=prompt_kwargs)
            if response.error or response.data is None:
                log.error(f"query: {query} failed to retrieve")
                log.error(f"error_message: {response.error}")
                log.error(f"raw_response: {response.raw_response}")
                log.error(f"response: {response.data}")
                retrieved_outputs.append(None)
                continue
            retrieved_outputs.append(response.data)
        return retrieved_outputs

    def __call__(
        self,
        query_or_queries: RetrieverInputType,
        top_k: Optional[int] = None,
    ) -> RetrieverOutputType:
        # query will be used
        return self.retrieve(query_or_queries, top_k)
