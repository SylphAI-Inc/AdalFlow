from typing import List, Optional, Any, Dict

from lightrag.core.component import Component
from lightrag.core.retriever import Retriever, RetrieverInputType, RetrieverOutputType
from lightrag.core.generator import Generator
from lightrag.core.model_client import ModelClient
from lightrag.core.default_prompt_template import DEFAULT_LIGHTRAG_SYSTEM_PROMPT
from lightrag.core.string_parser import ListParser
from lightrag.core.prompt_builder import Prompt

DEFAULT_LLM_AS_RETRIEVER_TASK_DESC = """Your are a retriever. Given a list of documents in the context, \
you will retrieve a list of {{top_k}} incices(int) of the documents that are most relevant to the query. You will output a list as follows:
[<id from the most relevent with top_k options>]
"""

DEFAULT_FORM_DOCUMENTS_STR_AS_CONTEXT_STR = r"""
{% for doc in documents %}
{{ loop.index - 1}}. {{doc}}
{% endfor %}
"""


class LLMRetriever(Retriever):
    r"""We need to configure the generator with model_client, model_kwargs, output_processors, and preset_prompt_kwargs"""

    def __init__(
        self,
        *,
        top_k: Optional[int] = 1,
        # the genearator kwargs
        model_client: ModelClient,
        model_kwargs: Dict[str, Any] = {},
        template: str = DEFAULT_LIGHTRAG_SYSTEM_PROMPT,
        preset_prompt_kwargs: Optional[Dict] = {},
        output_processors: Optional[Component] = ListParser(),
    ):
        super().__init__()
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=template,
            preset_prompt_kwargs=preset_prompt_kwargs,
            output_processors=output_processors,
        )
        print(f"generator: {self.generator}")
        self.top_k = top_k
        # the subprompt needs to be rendered at each call
        self.task_desc_prompt = Prompt(
            template=DEFAULT_LLM_AS_RETRIEVER_TASK_DESC,
            preset_prompt_kwargs={"top_k": top_k},
        )

    def build_index_from_documents(
        self,
        documents: List[str],
    ):
        """prepare the user query input for the retriever"""
        documents_to_use = documents.copy()
        self.index_prompt = Prompt(
            template=DEFAULT_FORM_DOCUMENTS_STR_AS_CONTEXT_STR,
            preset_prompt_kwargs={"documents": documents_to_use},
        )
        self.index_context_str = self.index_prompt()
        self.generator.prompt.update_preset_prompt_kwargs(
            context_str=self.index_context_str
        )

    def retrieve(
        self, query_or_queries: RetrieverInputType, top_k: Optional[int] = None
    ) -> RetrieverOutputType:
        # run the generator
        print(f"query_or_queries: {query_or_queries}")
        if top_k is None:
            top_k = self.top_k

        # render the task_desc_str
        task_desc_str = self.task_desc_prompt(top_k=top_k)

        prompt_kwargs = {
            "task_desc_str": task_desc_str,  # subprompt
        }

        response = self.generator(query_or_queries, prompt_kwargs=prompt_kwargs)
        return response

    def print_prompt(self):
        task_desc_str = self.task_desc_prompt()

        prompt_kwargs = {
            "task_desc_str": task_desc_str,
        }
        self.generator.prompt.print_prompt(**prompt_kwargs)

    def __call__(
        self,
        query_or_queries: RetrieverInputType,
        top_k: Optional[int] = None,
    ) -> RetrieverOutputType:
        # query will be used
        return self.retrieve(query_or_queries, top_k)
