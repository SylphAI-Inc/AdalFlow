from typing import List, Optional, Any, Callable, Dict, Union
from jinja2 import Template

from core.data_classes import RetrieverOutput, Document, RetrieverOutput
from core.retriever import Retriever, RetrieverInputType, RetrieverOutputType
from core.generator import Generator

DEFAULT_LLM_AS_RETRIEVER_TASK_DESC = r"""Your are a retriever. You will be given a list of documents, and you will retrieve documents in the following way:
{{retriever_standard_spec}}
You will output a list of {{top_k}} incices of the documents that are most relevant to the query. You will output a list as follows:
[..., ...]
"""

DEFAULT_FORM_DOCUMENTS_STR_AS_CONTEXT_STR = r"""
{% for doc in documents %}
{{ loop.index - 1}}. {{doc}}
{% endfor %}
"""


class LLMRetriever(Retriever):
    r"""We need to configure the generator with model_client, model_kwargs, output_processors, and preset_prompt_kwargs"""

    # set

    # initialize the
    def __init__(
        self,
        # retriever_standard_spec: str = DEFAULT_LLM_AS_RETRIEVER_TASK_DESC,
        generator_kwargs: Dict[str, Any] = {},  # all arguments for generator
        top_k: Optional[int] = 1,
    ):
        super().__init__()
        print(f"generator_kwargs: {generator_kwargs}")
        if (
            "preset_prompt_kwargs" in generator_kwargs
            and "task_desc_str" not in generator_kwargs["preset_prompt_kwargs"]
        ):
            generator_kwargs["preset_prompt_kwargs"][
                "task_desc_str"
            ] = DEFAULT_LLM_AS_RETRIEVER_TASK_DESC  # i want to set the retriever_standard_spec to the task_desc_str using jina preset?
        if "preset_prompt_kwargs" not in generator_kwargs:
            generator_kwargs["preset_prompt_kwargs"] = {
                "task_desc_str": DEFAULT_LLM_AS_RETRIEVER_TASK_DESC
            }

        # context_str: Optional[str] = None,
        generator_kwargs["preset_prompt_kwargs"][
            "context_str"
        ] = DEFAULT_FORM_DOCUMENTS_STR_AS_CONTEXT_STR
        self.generator = Generator(**generator_kwargs)
        print(f"generator: {self.generator}")
        # self.retriever_standard_spec = retriever_standard_spec
        self.top_k = top_k

    def build_index_from_documents(
        self,
        documents: List[Document],
        input_field_map_func: Callable[[Document], Any] = lambda x: x.text,
    ):
        """prepare the user query input for the retriever"""
        documents_to_use = [input_field_map_func(doc) for doc in documents]
        # form the index as the query_str in the default prompt
        # pass it to preset_prompt_kwargs documents
        template = Template(DEFAULT_FORM_DOCUMENTS_STR_AS_CONTEXT_STR)
        rendered_query_str = template.render(documents=documents_to_use)
        print(f"rendered_query_str: {rendered_query_str}")
        self.generator.preset_prompt_kwargs["context_str"] = rendered_query_str

        # self.generator.preset_prompt_kwargs["query_str"] = rendered_query_str

    def retrieve(
        self, query_or_queries: Union[str, List[str]], top_k: Optional[int] = None
    ) -> RetrieverOutputType:
        # run the generator
        print(f"query_or_queries: {query_or_queries}")
        # self.preset
        # render top_k
        if top_k is None:
            top_k = self.top_k
        # update the task_desc_str with the top_k
        template = Template(self.generator.preset_prompt_kwargs["task_desc_str"])
        rendered_query_str = template.render(top_k=top_k)
        self.generator.preset_prompt_kwargs["task_desc_str"] = rendered_query_str
        print(
            f"self.generator.preset_prompt_kwargs: {self.generator.preset_prompt_kwargs}"
        )
        response = self.generator(query_or_queries)
        # process it with list output parser
        return response

    def __call__(
        self,
        query_or_queries: RetrieverInputType,
        top_k: Optional[int] = None,
    ) -> RetrieverOutputType:
        # query will be used
        return self.retrieve(query_or_queries, top_k)


# step 1: pass a list of documents to the retriever with llm_retriever.build_index_from_documents(documents)
# step 2: call generator with super(Generator, self)(**generator_kwargs) to generate the response
# step 3: call retrieve with
# it is  a generator as retriever
# class LLMRetriever(Generator, Retriever):
#     r"""We need to configure the generator with model_client, model_kwargs, output_processors, and preset_prompt_kwargs"""

#     # set

#     # initialize the
#     def __init__(
#         self,
#         retriever_standard_spec: str = DEFAULT_LLM_AS_RETRIEVER_TASK_DESC,
#         generator_kwargs: Dict[str, Any] = {},  # all arguments for generator
#         retriever_kwargs: Dict[str, Any] = {},
#     ):
#         # TODO: set the default output parser to...ListOutputParser
#         if (
#             "preset_prompt_kwargs" in generator_kwargs
#             and "task_desc_str" not in generator_kwargs["preset_prompt_kwargs"]
#         ):
#             generator_kwargs["preset_prompt_kwargs"][
#                 "task_desc_str"
#             ] = retriever_standard_spec  # i want to set the retriever_standard_spec to the task_desc_str using jina preset?
#         if "preset_prompt_kwargs" not in generator_kwargs:
#             generator_kwargs["preset_prompt_kwargs"] = {
#                 "task_desc_str": retriever_standard_spec
#             }
#         # TODO: get default value for the generator_kwargs
#         Generator.__init__(self, **generator_kwargs)
#         Retriever.__init__(self, **retriever_kwargs)
#         self.retriever_standard_spec = retriever_standard_spec

#     def build_index_from_documents(
#         self,
#         documents: List[Document],
#         input_field_map_func: Callable[[Document], Any] = lambda x: x.text,
#     ):
#         """prepare the user query input for the retriever"""
#         documents_to_use = [input_field_map_func(doc) for doc in documents]
#         # form the index as the query_str in the default prompt
#         # pass it to preset_prompt_kwargs documents
#         template = Template(DEFAULT_FORM_DOCUMENTS_STR_AS_QUERY_STR)
#         rendered_query_str = template.render(documents=documents_to_use)
#         print(f"rendered_query_str: {rendered_query_str}")
#         self.preset_prompt_kwargs["query_str"] = rendered_query_str

#     # def combine()

#     def retrieve(
#         self, query_or_queries: Union[str, List[str]], top_k: Optional[int] = None
#     ) -> RetrieverOutputType:
#         # run the generator
#         print(f"query_or_queries: {query_or_queries}")
#         # self.preset
#         response = self.generate(query_or_queries)
#         # process it with list output parser
#         return response

#     def __call__(
#         self,
#         query_or_queries: RetrieverInputType,
#         top_k: Optional[int] = None,
#     ) -> RetrieverOutputType:
#         # query will be used
#         return self.retrieve(query_or_queries, top_k)
