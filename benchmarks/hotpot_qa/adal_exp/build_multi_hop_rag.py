"""We will use dspy's retriever to keep that the same and only use our generator and optimizer"""

import dspy
from typing import List, Optional
from dataclasses import dataclass, field

import adalflow as adal
from adalflow.optim.parameter import Parameter, ParameterType


from adalflow.core.retriever import Retriever

from benchmarks.hotpot_qa.adal_exp.build_vanilla_rag import DspyRetriever
from adalflow.utils.logger import printc
from adalflow.components.agent.react import ReActAgent

from adalflow.optim.grad_component import GradComponent

colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url="http://20.102.90.50:2017/wiki17_abstracts"
)

dspy.settings.configure(rm=colbertv2_wiki17_abstracts)


# task pipeline


# dspy format
# Follow the following format.
# Context: may contain relevant facts
# Question: ${question}
# Reasoning: Let's think step by step in order to ${produce the query}. We ...
# Query: ${query}
@dataclass
class QueryRewritterData(adal.DataClass):
    reasoning: str = field(
        metadata={"desc": "The reasoning to produce the query"},
    )
    query: str = field(
        metadata={"desc": "The query you produced"},
    )

    __output_fields__ = ["reasoning", "query"]


query_template = """<START_OF_SYSTEM_PROMPT>
{{task_desc_str}}

{{output_format_str}}
{# Few shot demos #}
{% if few_shot_demos is not none %}
Here are some examples:
{{few_shot_demos}}
{% endif %}
<END_OF_SYSTEM_PROMPT>
<START_OF_USER>
Question: {{question}}
{% if last_query is not none %}
Last Query: {{last_query}}
{% endif %}
{% if context is not none %}
Context from last search query: {{context}}
{% endif %}
<END_OF_USER>
"""


@dataclass
class QueriesOutput(adal.DataClass):
    data: str = field(
        metadata={"desc": "The joined queries"},
    )
    id: str = field(
        metadata={"desc": "The id of the output"},
    )


# class DeduplicateList(adal.GradComponent):
#     def __init__(self):
#         super().__init__()

#     def call(
#         self, exisiting_list: List[str], new_list: List[str], id: str = None
#     ) -> List[str]:

#         seen = set()
#         return [x for x in exisiting_list + new_list if not (x in seen or seen.add(x))]

#     def backward(self, *args, **kwargs):

#         printc(f"DeduplicateList backward: {args}", "yellow")
#         return super().backward(*args, **kwargs)


class CombineList(GradComponent):
    def __init__(
        self,
        name="CombineRetrieverOut",
        desc="combines two lists and deduplicate with set",
    ):
        super().__init__(name=name, desc=desc)

    def call(
        self,
        context_1: adal.RetrieverOutput,
        context_2: adal.RetrieverOutput,
        id: str = None,
    ) -> List[str]:

        seen = set()
        lists_1 = context_1.documents
        lists_2 = context_2.documents
        combined = [x for x in lists_1 + lists_2 if not (x in seen or seen.add(x))]

        output = adal.RetrieverOutput(
            id=id,
            # query=f"query 1: {context_1.query}, query 2: {context_2.query}",
            query=[context_1.query, context_2.query],
            documents=combined,
            doc_indices=[],
        )
        return output


class CombineQueries(GradComponent):
    def __init__(
        self,
        name="CombineTwoQueries using ','",
        desc="combines two queries for evaluation",
    ):
        super().__init__(name=name, desc=desc)

    def call(
        self,
        q_1: str,
        q_2: str,
        id: str = None,
    ) -> QueriesOutput:

        value = f"{q_1}, {q_2}"

        output = QueriesOutput(data=value, id=id)

        return output


query_generator_task_desc = """Write a simple search query that will help answer a complex question.

You will receive a context(may contain relevant facts) and a question.
Think step by step."""


task_desc_str = """
You will receive an original question, last search query, and the retrieved context from the last search query.
Write the next search query to help retrieve all relevant context to answer the original question.
Think step by step."""

task_desc_str_system_finetuned = """
Write a search query to identify key information step by step. Begin by extracting names or entities directly referenced in the question. Use retrieved data to iteratively refine subsequent queries, targeting specific attributes such as filmographies, roles, or numerical criteria (e.g., number of movies or TV shows). Adjust the query dynamically based on gaps or ambiguities in retrieved results.
"""

task_desc_system_finedtuned_separately = [
    "Write a search query that extracts the key entity or fact required to begin answering the question. Focus on identifying specific names, titles, or roles directly referenced in the question. The query should aim to retrieve precise and relevant details (e.g., the name of a person, cast members of a movie, or associated facts) to refine understanding of the question.",
    "Based on the retrieved results, refine the search query to target detailed information that resolves the question. Use retrieved entities or partial answers to adjust the query dynamically. If gaps or ambiguities remain, incorporate criteria from the original question (e.g., specific numbers, attributes, or context) to improve precision and relevance.",
]


class MultiHopRetrieverCycle(adal.Retriever):
    def __init__(self, model_client, model_kwargs, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.passages_per_hop = passages_per_hop
        self.max_hops = max_hops

        self.data_parser = adal.DataClassParser(
            data_class=QueryRewritterData, return_data_class=True, format_type="json"
        )

        # only one generator which will be used in a loop, called max_hops times
        self.query_generator: adal.Generator = adal.Generator(
            name="query_generator",
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs={
                # "few_shot_demos": Parameter(
                #     name="few_shot_demos",
                #     data=None,
                #     role_desc="To provide few shot demos to the language model",
                #     requires_opt=True,
                #     param_type=ParameterType.DEMOS,
                # ),
                "task_desc_str": Parameter(
                    name="task_desc_str",
                    data=task_desc_str,
                    # data=task_desc_str_system_finetuned,
                    # data=task_desc_system_finedtuned_separately[0],
                    role_desc="Task description for the language model. Used together with \
                    the following template: \
                    Question: {{question}} \
{% if last_query is not none %} \
Last Query: {{last_query}}\
{% endif %}\
{% if context is not none %}\
Context from last search query: {{context}}\
{% endif %}",
                    requires_opt=True,
                    param_type=ParameterType.PROMPT,
                ),
                "output_format_str": self.data_parser.get_output_format_str(),
            },
            template=query_template,
            output_processors=self.data_parser,
            use_cache=True,
        )

        self.retriever = DspyRetriever(top_k=passages_per_hop)
        self.combine_list = CombineList()

    @staticmethod
    def context_to_str(context: List[str]) -> str:
        return "\n".join(context)

    def call(self, *, input: str, id: str = None) -> List[adal.RetrieverOutput]:
        # assemble the foundamental building blocks
        out = self.forward(input=input, id=id)

        if not isinstance(out, adal.Parameter):
            raise ValueError("The output should be a parameter")

        return out.data  # or full response its up to users

    def forward(self, *, input: str, id: str = None) -> adal.Parameter:
        context = []
        # 1. make question a parameter as generator does not have it yet
        # can create the parameter at the leaf, but not the intermediate nodes
        question_param = adal.Parameter(
            name="question",
            data=input,
            role_desc="The question to be answered",
            requires_opt=False,
            param_type=ParameterType.INPUT,
        )
        contexts = []
        last_query = None

        for i in range(self.max_hops):

            gen_out = self.query_generator.forward(
                prompt_kwargs={
                    "context": context,
                    "question": question_param,
                    "last_query": last_query,
                    # "task_desc_str": task_desc_system_finedtuned_separately[
                    #     i
                    # ],  # replace this at runtime
                },
                id=id,
            )
            # prompt_kwargs = {
            #     "context": context,
            #     "question": question_param,
            #     "last_query": last_query,
            # }
            # prompt = self.query_generator.get_prompt(**prompt_kwargs)
            # printc(f"prompt: {prompt}", "yellow")

            # printc(f"query {i}: {gen_out.data.data.query}", "yellow")
            # extract the query from the generator output
            success_map_fn = lambda x: (  # noqa E731
                x.data.data.query
                if x.data and x.data.data and x.data.data.query
                else (x.data.raw_response if x.data and x.data.raw_response else None)
            )
            # print(f"query {i}: {success_map_fn(gen_out)}")

            gen_out.add_successor_map_fn(
                successor=self.retriever, map_fn=success_map_fn
            )
            # printc(f"before retrieve_out: {success_map_fn(gen_out)}", "yellow")

            # retrieve the passages
            retrieve_out: adal.Parameter = self.retriever.forward(input=gen_out, id=id)
            # printc(f"retrieve_out: {retrieve_out}", "yellow")

            retrieve_out.data_in_prompt = lambda x: {
                "query": x.data.query,
                "documents": x.data.documents,
            }
            if i + 1 < self.max_hops:
                last_query = gen_out

                last_query.add_successor_map_fn(
                    successor=self.query_generator, map_fn=success_map_fn
                )

            def retrieve_out_map_fn(x: adal.Parameter):
                return x.data.documents if x.data and x.data.documents else []

            # add the map function to the retrieve_out
            retrieve_out.add_successor_map_fn(
                successor=self.deduplicater, map_fn=retrieve_out_map_fn
            )
            context = retrieve_out
            if i + 1 < self.max_hops:
                context.add_successor_map_fn(
                    successor=self.query_generator, map_fn=retrieve_out_map_fn
                )

            contexts.append(context)

        contexts[0].add_successor_map_fn(
            successor=self.combine_list, map_fn=lambda x: x.data
        )
        contexts[1].add_successor_map_fn(
            successor=self.combine_list, map_fn=lambda x: x.data
        )

        context_sum = self.combine_list.forward(contexts[0], contexts[1])
        return context_sum


# task_desc_str = """Write a simple search query that will help answer a complex question.

# You will receive a context(may contain relevant facts) and a question.
# Think step by step."""


trained_task_desc_strs = [
    "You are tasked with formulating precise search queries using the original question, last search query, and its retrieved context. Prioritize identifying, emphasizing, and explicitly including all crucial entities, relationships, and geographical details mentioned in the question. Ensure comprehensive retrieval by focusing on key elements such as specific individuals (e.g., 'Kyrie Irving'), roles, or contextual details required for accuracy. Demonstrate reasoning by cross-referencing multiple sources and provide clear examples where necessary. Adapt queries to capture all nuances effectively for improved relevance and accuracy. Think step by step.",
    "You will receive an original question, the last search query, and the retrieved context from that search. Write the next search query to ensure comprehensive retrieval of all relevant context needed to answer the original question. Emphasize identifying, precisely including, and verifying specific key entities, historical events, and factual names directly linked to the question within the context. Explicitly use the context to confirm and match critical entities to improve recall and ensure consistency with the targeted entities. Avoid irrelevant inclusions or false positives by cross-referencing data and verifying alignment accurately. Think step by step.",
]

trained_task_desc_strs = [
    "You will receive an original question, last search query, and the retrieved context from the last search query. Identify key entities, explicitly named individuals, and specific versions (e.g., specific film versions) in the original question to ensure comprehensive and focused retrieval. Craft a refined search query to help retrieve relevant context, prioritizing connections and biographical details needed. Think step by step.",
    "You will receive an original question, last search query, and the retrieved context from the last search query. Analyze both the question and context to craft the next search query. Focus on all pertinent entities, especially notable individuals, mentioned in the question and context to ensure comprehensive coverage. Think step by step.",
]

few_shot_demos = [
    "reasoning: The question is asking for the individual who defeated Sander Levin in\n  a specific election, the Michigan gubernatorial election of 1970. I need to determine\n  who his opponent was and who won that election. Hence, I should focus the search\n  on the Michigan gubernatorial election of 1970, Sander Levin, and the name of the\n  winner.\nquery: Michigan gubernatorial election 1970 winner Sander Levin\n\nquestion: What is the name of this American law firm headquartered in Little Rock,\n  Arkansas, which was co-founded by Robert Crittenden?\nanswer: Rose Law Firm",
    "reasoning: The context provides information about Kirk Humphreys, the chairman of\n  The Humphreys Company, and his birth date as September 13, 1950. It also mentions\n  that he lost in a primary to former Congressman Tom Coburn, who is a medical doctor.\n  To determine who is older, we need to find the birth date of Tom Coburn.\nquery: Tom Coburn birth date\n\nquestion: In which century was football introduced to this region represented by FC\n  Espanya de Barcelona?\nanswer: 19th century",
]

manual_task_desc_strs = [
    "You will receive an question that requires 2 retrieveal steps to have enough context to answer. \
    You are the first step, write a simple search query to retrieve the first part of the context. \
    Think step by step.",
    "You will receive an original question, last search query, and the retrieved context from the last search query. Write the next search query to help retrieve all relevant context to answer the original question. Think step by step.",
]


# task_desc_str = """ You are a query assistant that helps search all relevant context to answer a multi-hop question.

# You will a question, and existing context(may contain relevant facts along with its sub-questions).
# Write a new simple search query to help retrieve the relevant context to answer the question.
# Think step by step."""


class MultiHopRetriever(adal.Component):
    def __init__(self, model_client, model_kwargs, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.passages_per_hop = passages_per_hop
        self.max_hops = max_hops

        self.data_parser = adal.DataClassParser(
            data_class=QueryRewritterData, return_data_class=True, format_type="json"
        )

        self.query_generators: adal.ComponentList[adal.Generator] = adal.ComponentList()
        self.retrievers: List[Retriever] = []
        self.deduplicaters: List[adal.GradComponent] = []

        for i in range(self.max_hops):
            self.query_generators.append(
                adal.Generator(
                    name=f"query_generator_{i}",
                    model_client=model_client,
                    model_kwargs=model_kwargs,
                    prompt_kwargs={
                        # "few_shot_demos": Parameter(
                        #     name=f"few_shot_demos_{i}",
                        #     # data=few_shot_demos[i],
                        #     data=None,
                        #     role_desc="To provide few shot demos to the language model",
                        #     requires_opt=True,
                        #     param_type=ParameterType.DEMOS,
                        # ),
                        "task_desc_str": Parameter(
                            name="task_desc_str",
                            data=task_desc_str,
                            # data=manual_task_desc_strs[i],
                            role_desc=f"""Task description for the {i+1}th language model."""
                            + "Used together with the following template: \
Question: {{question}} \
{% if last_query is not none %} \
Last Query: {{last_query}}\
{% endif %}\
{% if context is not none %}\
Context from last search query: {{context}}\
{% endif %}",
                            requires_opt=True,
                            param_type=ParameterType.PROMPT,
                        ),
                        "output_format_str": self.data_parser.get_output_format_str(),
                    },
                    template=query_template,
                    output_processors=self.data_parser,
                    use_cache=True,
                )
            )
            self.retrievers.append(DspyRetriever(top_k=passages_per_hop))

        self.combine_list = CombineList()
        self.combine_queries = CombineQueries()

    @staticmethod
    def context_to_str(context: List[str]) -> str:
        return "\n".join(context)

    @staticmethod
    def deduplicate(seq: list[str]) -> list[str]:
        """
        Source: https://stackoverflow.com/a/480227/1493011
        """

        seen = set()
        return [x for x in seq if not (x in seen or seen.add(x))]

    def call(self, *, input: str, id: str = None) -> adal.RetrieverOutput:
        context = []
        queries: List[str] = []
        last_query = None
        for i in range(self.max_hops):
            gen_out = self.query_generators[i](
                prompt_kwargs={
                    "context": context,
                    "question": input,
                    "last_query": last_query,
                },
                id=id,
            )

            query = gen_out.data.query if gen_out.data and gen_out.data.query else input

            retrieve_out = self.retrievers[i](input=query, id=id)

            passages = retrieve_out.documents
            context = self.deduplicate(context + passages)
            queries.append(query)
            last_query = query
        out = adal.RetrieverOutput(
            documents=context, query=queries, doc_indices=[], id=id
        )
        return out

    def forward(self, *, input: str, id: str = None) -> adal.Parameter:

        queries: List[str] = []

        context = []
        last_query = None
        contexts: List[Parameter] = []

        for i in range(self.max_hops):
            gen_out: Parameter = self.query_generators[i].forward(
                prompt_kwargs={
                    "context": context,
                    "last_query": last_query,
                    "question": adal.Parameter(
                        name="question",
                        data=input,
                        role_desc="The question to be answered",
                        requires_opt=False,
                        param_type=ParameterType.INPUT,
                    ),
                },
                id=id,
            )

            success_map_fn = lambda x: (  # noqa E731
                x.data.data.query
                if x.data and x.data.data and x.data.data.query
                else (x.data.raw_response if x.data and x.data.raw_response else None)
            )
            # printc(f"query {i}: {success_map_fn(gen_out)}")

            queries.append(success_map_fn(gen_out))

            gen_out.add_successor_map_fn(
                successor=self.retrievers[i], map_fn=success_map_fn
            )

            if success_map_fn(gen_out) is None:
                raise ValueError(f"The query is None, please check the generator {i}")

            retrieve_out = self.retrievers[i].forward(input=gen_out, id=id)

            def retrieve_out_map_fn(x: adal.Parameter):
                return x.data.documents if x.data and x.data.documents else []

            retrieve_out.data_in_prompt = lambda x: {
                "query": x.data.query,
                "documents": x.data.documents,
            }
            context = retrieve_out
            if i + 1 < self.max_hops:
                context.add_successor_map_fn(
                    successor=self.query_generators[i + 1], map_fn=retrieve_out_map_fn
                )
                last_query = success_map_fn(gen_out)
            contexts.append(retrieve_out)

        contexts[0].add_successor_map_fn(
            successor=self.combine_list, map_fn=lambda x: x.data
        )
        contexts[1].add_successor_map_fn(
            successor=self.combine_list, map_fn=lambda x: x.data
        )
        contexts_sum = self.combine_list.forward(
            context_1=contexts[0], context_2=contexts[1]
        )
        contexts_sum.data_in_prompt = lambda x: {
            "query": x.data.query,
            "documents": x.data.documents,
        }

        return contexts_sum

    # TODO: might need to support multiple output parameters
    def forward2(self, *, input: str, id: str = None) -> List[adal.Parameter]:
        r"""Experiment multiple output parameters for multiple evaluation."""
        # assemble the foundamental building blocks
        printc(f"question: {input}", "yellow")

        queries: List[adal.Parameter] = []

        context = []
        last_query = None
        contexts: List[Parameter] = []

        for i in range(self.max_hops):
            gen_out: Parameter = self.query_generators[i].forward(
                prompt_kwargs={
                    "context": context,
                    "last_query": last_query,
                    "question": adal.Parameter(
                        name="question",
                        data=input,
                        role_desc="The question to be answered",
                        requires_opt=False,
                        param_type=ParameterType.INPUT,
                    ),
                },
                id=id,
            )

            success_map_fn = lambda x: (  # noqa E731
                x.data.data.query
                if x.data and x.data.data and x.data.data.query
                else (x.data.raw_response if x.data and x.data.raw_response else None)
            )
            # printc(f"query {i}: {success_map_fn(gen_out)}")

            # queries.append(success_map_fn(gen_out))
            queries.append(gen_out)

            gen_out.add_successor_map_fn(
                successor=self.retrievers[i], map_fn=success_map_fn
            )

            if success_map_fn(gen_out) is None:
                raise ValueError(f"The query is None, please check the generator {i}")

            retrieve_out = self.retrievers[i].forward(input=gen_out, id=id)

            def retrieve_out_map_fn(x: adal.Parameter):
                return x.data.documents if x.data and x.data.documents else []

            # print(f"retrieve_out: {retrieve_out}")

            # retrieve_out.add_successor_map_fn(
            #     successor=self.deduplicaters[i], map_fn=retrieve_out_map_fn
            # )
            context = retrieve_out
            if i + 1 < self.max_hops:
                context.add_successor_map_fn(
                    successor=self.query_generators[i + 1], map_fn=retrieve_out_map_fn
                )

            # context = self.deduplicaters[i].forward(
            #     exisiting_list=context, new_list=retrieve_out
            # )
            contexts.append(retrieve_out)
            if i + 1 < self.max_hops:
                retrieve_out.add_successor_map_fn(
                    successor=self.query_generators[i + 1], map_fn=retrieve_out_map_fn
                )

                last_query = success_map_fn(gen_out)

        queries[0].add_successor_map_fn(
            successor=self.combine_queries, map_fn=lambda x: x.data.data.query
        )
        queries[1].add_successor_map_fn(
            successor=self.combine_queries, map_fn=lambda x: x.data.data.query
        )
        combined_queries = self.combine_queries.forward(q_1=queries[0], q_2=queries[1])
        printc(f"queries: {combined_queries.data}", "yellow")
        return combined_queries


from benchmarks.hotpot_qa.adal_exp.build_vanilla_rag import (
    VanillaRAG,
)


class MultiHopRAG(VanillaRAG):
    def __init__(
        self, passages_per_hop=3, max_hops=2, model_client=None, model_kwargs=None
    ):
        super().__init__(
            passages_per_hop=passages_per_hop,
            model_client=model_client,
            model_kwargs=model_kwargs,
        )
        self.retriever = MultiHopRetriever(
            model_client=model_client,
            model_kwargs=model_kwargs,
            passages_per_hop=passages_per_hop,
            max_hops=max_hops,
        )
        # update the parameters to untainable
        # for name, param in self.llm.named_parameters():
        #     param.requires_opt = False
        #     printc(f"param: {name} requires_opt: {param.requires_opt}", "yellow")


class MultiHopRAGCycle(VanillaRAG):
    def __init__(
        self, passages_per_hop=3, max_hops=2, model_client=None, model_kwargs=None
    ):
        super().__init__(
            passages_per_hop=passages_per_hop,
            model_client=model_client,
            model_kwargs=model_kwargs,
        )
        self.retriever = MultiHopRetrieverCycle(
            model_client=model_client,
            model_kwargs=model_kwargs,
            passages_per_hop=passages_per_hop,
            max_hops=max_hops,
        )


# TODO: agent needs storage for the context instead of all in the step history.
class AgenticRAG(adal.GradComponent):
    def __init__(self, model_client, model_kwargs):
        super().__init__()

        self.dspy_retriever = DspyRetriever(top_k=2)
        # self.llm_parser = adal.DataClassParser(
        #     data_class=AnswerData, return_data_class=True, format_type="json"
        # )
        # self.llm = adal.Generator(
        #     model_client=model_client,
        #     model_kwargs=model_kwargs,
        #     template=answer_template,
        #     prompt_kwargs={
        #         "task_desc_str": adal.Parameter(
        #             data=task_desc_str,
        #             role_desc="Task description for the language model",
        #             param_type=adal.ParameterType.PROMPT,
        #             requires_opt=True,
        #         ),
        #         "few_shot_demos": adal.Parameter(
        #             data=None,
        #             requires_opt=None,
        #             role_desc="To provide few shot demos to the language model",
        #             param_type=adal.ParameterType.DEMOS,
        #         ),
        #         "output_format_str": self.llm_parser.get_output_format_str(),
        #     },
        #     output_processors=self.llm_parser,
        # )

        # self.context = []

        def dspy_retriever_as_tool(
            input: str,
            # context_variables: Dict,
            id: Optional[str] = None,
        ) -> List[str]:
            r"""Retrieves the top 2 passages from using input as the query.
            Ensure you get all the context to answer the original question.
            """
            output = self.dspy_retriever(input=input, id=id)
            parsed_output = output
            if isinstance(output, adal.Parameter):
                parsed_output = output.data.documents
                return parsed_output
            documents = parsed_output.documents
            # if context_variables:
            #     context_variables["context"].extend(documents)
            return documents

        # def generator_as_tool(
        #     input: str,
        #     context_variables: Dict,
        #     id: Optional[str] = None,
        # ) -> str:
        #     r"""Generates the answer to the question(input) and the context from the context_variables(Dict).
        #     Example: generator_as_tool(original question, context_variables=context_variables)

        #     YOU MUST call generator_as_tool once to produce the final answer.
        #     """
        #     context = context_variables["context"]
        #     # print(f"context: {context}")
        #     output = self.llm(
        #         prompt_kwargs={"question": input, "context": context}, id=id
        #     )
        #     return output

        from adalflow.core.func_tool import FunctionTool

        tools = [
            FunctionTool(self.dspy_retriever.__call__, component=self.dspy_retriever),
            # FunctionTool(generator_as_tool, component=self.llm),
        ]  # NOTE: agent is not doing well to call component methods at this moment

        tools = [
            FunctionTool(dspy_retriever_as_tool, component=self.dspy_retriever),
            # FunctionTool(generator_as_tool, component=self.llm),
        ]

        self.agent = ReActAgent(
            max_steps=3,
            add_llm_as_fallback=False,
            tools=tools,
            model_client=model_client,
            model_kwargs=model_kwargs,
            context_variables=None,
        )

    def forward(self, *args, **kwargs) -> Parameter:
        return self.bicall(*args, **kwargs)

    def call(self, *args, **kwargs):
        return self.bicall(*args, **kwargs)

    def bicall(self, input: str, id: str = None) -> str:
        out = self.agent(input=input, id=id)
        if isinstance(out, adal.Parameter):
            return out
        return out  # .observation ReactOutput
        # if isinstance(out, adal.Parameter):
        #     return out.data[-1].observation
        # return out[-1].observation


def test_multi_hop_retriever():

    from use_cases.config import (
        gpt_3_model,
    )

    multi_hop_retriever = MultiHopRetriever(
        **gpt_3_model,
        passages_per_hop=3,
        max_hops=2,
    )

    question = "How many storeys are in the castle that David Gregory inherited?"
    print(f"multi_hop_retriever: {multi_hop_retriever}")
    return
    # eval mode
    output = multi_hop_retriever.call(input=question, id="1")
    print(output)

    # train mode
    multi_hop_retriever.train()
    output = multi_hop_retriever.forward(input=question, id="1")
    print(output)
    output.draw_graph()


def test_multi_hop_retriever_cycle():

    from use_cases.config import (
        gpt_3_model,
    )

    multi_hop_retriever = MultiHopRetrieverCycle(
        **gpt_3_model,
        passages_per_hop=3,
        max_hops=2,
    )

    question = "How many storeys are in the castle that David Gregory inherited?"

    # eval mode
    output = multi_hop_retriever.call(input=question, id="1")
    print(output)

    # train mode
    multi_hop_retriever.train()
    output = multi_hop_retriever.forward(input=question, id="1")
    print(output)
    output.draw_graph()
    output.draw_output_subgraph()
    output.draw_component_subgraph()


def test_agent_rag():

    from use_cases.config import (
        gpt_3_model,
    )

    task = AgenticRAG(
        **gpt_3_model,
    )
    print(task)

    question = "How many storeys are in the castle that David Gregory inherited?"

    task.train()
    output = task(input=question, id="1")
    print(output.data)
    output.draw_graph()

    # output =
    # print(output)
    # output.draw_graph()
    # output.draw_output_subgraph()
    # output.draw_component_subgraph()

    # task.eval()
    # output = task(input=question, id="1")


def test_multi_hop_retriever2():

    from use_cases.config import (
        gpt_3_model,
    )

    multi_hop_retriever = MultiHopRetriever(
        **gpt_3_model,
        passages_per_hop=3,
        max_hops=2,
    )

    question = "How many storeys are in the castle that David Gregory inherited?"

    # eval mode
    # output = multi_hop_retriever.call(question=question, id="1")
    # print(output)

    # train mode
    multi_hop_retriever.train()
    output = multi_hop_retriever.forward(input=question, id="1")
    # print(output)
    output.draw_graph(full_trace=True)

    # multi_hop_retriever.eval()
    # output = multi_hop_retriever.call(input=question, id="1")
    # print(output)


def test_multi_hop_rag():

    from use_cases.config import (
        gpt_3_model,
    )

    adal.get_logger(level="DEBUG")

    task = MultiHopRAG(
        **gpt_3_model,
        passages_per_hop=3,
        max_hops=2,
    )
    print(f"task: {task}")

    for name, comp in task.named_components():

        if isinstance(comp, adal.Generator):
            print(f"name: {name}")
            print(f"comp: {comp }")
    return

    # test the retriever

    question = "How many storeys are in the castle that David Gregory inherited?"

    task.train()

    # id = "1"

    # retriever_out = task.retriever(input=question, id=id)

    # print(f"retriever_out: {retriever_out}")

    # test the forward function
    generator_out = task.forward(question=question, id="1")
    print(f"generator_out: {generator_out}")

    generator_out.draw_graph()

    # task.eval()
    # generator_out = task.call(question=question, id="1")
    # print(f"generator_out: {generator_out}")


if __name__ == "__main__":
    ### Try the minimum effort to test on any task

    # get_logger(level="DEBUG")
    # test_multi_hop_retriever()
    # test_multi_hop_retriever2()

    # test_multi_hop_retriever_cycle()
    # test_multi_hop_rag()
    test_agent_rag()
