"""We will use dspy's retriever to keep that the same and only use our generator and optimizer"""

import dspy
from typing import List, Optional, Dict
from dataclasses import dataclass, field

import adalflow as adal
from adalflow.optim.parameter import Parameter, ParameterType


from adalflow.core.retriever import Retriever

from benchmarks.hotpot_qa.adal_exp.build_vanilla_rag import DspyRetriever
from adalflow.utils.logger import printc

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
Context: {{context}}
Question: {{question}}
<END_OF_USER>
"""


class DeduplicateList(adal.GradComponent):
    def __init__(self):
        super().__init__()

    def call(
        self, exisiting_list: List[str], new_list: List[str], id: str = None
    ) -> List[str]:

        seen = set()
        return [x for x in exisiting_list + new_list if not (x in seen or seen.add(x))]

    def backward(self, *args, **kwargs):

        printc(f"DeduplicateList backward: {args}", "yellow")
        return super().backward(*args, **kwargs)


# User customize an auto-grad operator
# Need this to be a GradComponent


# NOTE: deprecated
# class MultiHopRetriever(adal.Retriever):
#     def __init__(self, model_client, model_kwargs, passages_per_hop=3, max_hops=2):
#         super().__init__()

#         self.passages_per_hop = passages_per_hop
#         self.max_hops = max_hops

#         self.data_parser = adal.DataClassParser(
#             data_class=QueryRewritterData, return_data_class=True, format_type="json"
#         )

#         # Grad Component
#         self.query_generators: List[adal.Generator] = []
#         for i in range(self.max_hops):
#             self.query_generators.append(
#                 adal.Generator(
#                     name=f"query_generator_{i}",
#                     model_client=model_client,
#                     model_kwargs=model_kwargs,
#                     prompt_kwargs={
#                         "few_shot_demos": Parameter(
#                             name="few_shot_demos_1",
#                             data=None,
#                             role_desc="To provide few shot demos to the language model",
#                             requires_opt=True,
#                             param_type=ParameterType.DEMOS,
#                         ),
#                         "task_desc_str": Parameter(
#                             name="task_desc_str",
#                             data="""Write a simple search query that will help answer a complex question.

# You will receive a context(may contain relevant facts) and a question.
# Think step by step.""",
#                             role_desc="Task description for the language model",
#                             requires_opt=True,
#                             param_type=ParameterType.PROMPT,
#                         ),
#                         "output_format_str": self.data_parser.get_output_format_str(),
#                     },
#                     template=query_template,
#                     output_processors=self.data_parser,
#                     use_cache=True,
#                 )
#             )
#         self.retriever = DspyRetriever(top_k=passages_per_hop)
#         self.deduplicater = DeduplicateList()

#     @staticmethod
#     def context_to_str(context: List[str]) -> str:
#         return "\n".join(context)

#     @staticmethod
#     def deduplicate(seq: list[str]) -> list[str]:
#         """
#         Source: https://stackoverflow.com/a/480227/1493011
#         """

#         seen = set()
#         return [x for x in seq if not (x in seen or seen.add(x))]

#     def call(self, *, question: str, id: str = None) -> adal.RetrieverOutput:
#         context = []
#         print(f"question: {question}")
#         for i in range(self.max_hops):
#             gen_out = self.query_generators[i](
#                 prompt_kwargs={
#                     "context": self.context_to_str(context),
#                     "question": question,
#                 },
#                 id=id,
#             )

#             query = gen_out.data.query if gen_out.data and gen_out.data.query else None

#             print(f"query {i}: {query}")

#             retrieve_out = self.retriever.call(input=query)
#             passages = retrieve_out[0].documents
#             context = self.deduplicate(context + passages)
#         out = [adal.RetrieverOutput(documents=context, query=query, doc_indices=[])]
#         return out

#     def forward(self, *, question: str, id: str = None) -> adal.Parameter:
#         # assemble the foundamental building blocks
#         context = []
#         print(f"question: {question}")
#         # 1. make question a parameter as generator does not have it yet
#         # can create the parameter at the leaf, but not the intermediate nodes
#         question_param = adal.Parameter(
#             name="question",
#             data=question,
#             role_desc="The question to be answered",
#             requires_opt=True,
#             param_type=ParameterType.INPUT,
#         )
#         context_param = adal.Parameter(
#             name="context",
#             data=context,
#             role_desc="The context to be used for the query",
#             requires_opt=True,
#             param_type=ParameterType.INPUT,
#         )
#         context_param.add_successor_map_fn(
#             successor=self.query_generators[0],
#             map_fn=lambda x: self.context_to_str(x.data),
#         )

#         for i in range(self.max_hops):

#             gen_out = self.query_generators[i].forward(
#                 prompt_kwargs={
#                     "context": context_param,
#                     "question": question_param,
#                 },
#                 id=id,
#             )

#             success_map_fn = lambda x: (  # noqa E731
#                 x.full_response.data.query
#                 if x.full_response
#                 and x.full_response.data
#                 and x.full_response.data.query
#                 else None
#             )
#             print(f"query {i}: {success_map_fn(gen_out)}")

#             gen_out.add_successor_map_fn(
#                 successor=self.retriever, map_fn=success_map_fn
#             )

#             retrieve_out = self.retriever.forward(input=gen_out)

#             def retrieve_out_map_fn(x: adal.Parameter):
#                 return x.data[0].documents if x.data and x.data[0].documents else []

#             print(f"retrieve_out: {retrieve_out}")

#             retrieve_out.add_successor_map_fn(
#                 successor=self.deduplicater, map_fn=retrieve_out_map_fn
#             )

#             context_param = self.deduplicater.forward(
#                 exisiting_list=context_param, new_list=retrieve_out
#             )

#         context_param.param_type = ParameterType.RETRIEVER_OUTPUT

#         return context_param

query_generator_task_desc = """Write a simple search query that will help answer a complex question.

You will receive a context(may contain relevant facts) and a question.
Think step by step."""


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
                "few_shot_demos": Parameter(
                    name="few_shot_demos",
                    data=None,
                    role_desc="To provide few shot demos to the language model",
                    requires_opt=False,
                    param_type=ParameterType.DEMOS,
                ),
                "task_desc_str": Parameter(
                    name="task_desc_str",
                    data=query_generator_task_desc,
                    role_desc="Task description for the language model",
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
        self.deduplicater = DeduplicateList()

    @staticmethod
    def context_to_str(context: List[str]) -> str:
        return "\n".join(context)

    def call(self, *, input: str, id: str = None) -> List[adal.RetrieverOutput]:
        # assemble the foundamental building blocks
        printc(f"question: {input}", "yellow")
        out = self.forward(input=input, id=id)

        if not isinstance(out, adal.Parameter):
            raise ValueError("The output should be a parameter")

        return out.data  # or full response its up to users

    def forward(self, *, input: str, id: str = None) -> adal.Parameter:
        # assemble the foundamental building blocks
        context = []
        # queries: List[str] = []
        print(f"question: {input}")
        # 1. make question a parameter as generator does not have it yet
        # can create the parameter at the leaf, but not the intermediate nodes
        question_param = adal.Parameter(
            name="question",
            data=input,
            role_desc="The question to be answered",
            requires_opt=False,
            param_type=ParameterType.INPUT,
        )
        # context_param = adal.Parameter(
        #     name="context",
        #     data=context,
        #     role_desc="The context to be used for the query",
        #     requires_opt=True,
        #     param_type=ParameterType.INPUT,
        # )
        # context_param.add_successor_map_fn(
        #     successor=self.query_generator,
        #     map_fn=lambda x: self.context_to_str(x.data),
        # )

        for i in range(self.max_hops):

            gen_out = self.query_generator.forward(
                prompt_kwargs={
                    "context": context,
                    "question": question_param,
                },
                id=id,
            )
            # extract the query from the generator output
            success_map_fn = lambda x: (  # noqa E731
                x.full_response.data.query
                if x.full_response
                and x.full_response.data
                and x.full_response.data.query
                else (
                    x.full_response.raw_response
                    if x.full_response and x.full_response.raw_response
                    else None
                )
            )
            print(f"query {i}: {success_map_fn(gen_out)}")

            gen_out.add_successor_map_fn(
                successor=self.retriever, map_fn=success_map_fn
            )

            # retrieve the passages
            retrieve_out: adal.Parameter = self.retriever.forward(input=gen_out, id=id)

            def retrieve_out_map_fn(x: adal.Parameter):
                return x.data[0].documents if x.data and x.data[0].documents else []

            # add the map function to the retrieve_out
            retrieve_out.add_successor_map_fn(
                successor=self.deduplicater, map_fn=retrieve_out_map_fn
            )

            # combine the context + deduplicated passages
            context = self.deduplicater.forward(
                exisiting_list=context, new_list=retrieve_out, id=id
            )

        context.param_type = ParameterType.RETRIEVER_OUTPUT
        context.requires_opt = True
        # used as the final outptu

        # convert the context to the retriever output
        def context_to_retrover_output(x):
            return [
                adal.RetrieverOutput(
                    documents=x.data,
                    query=[input] + [success_map_fn(gen_out)],
                    doc_indices=[],
                )
            ]

        context.data = context_to_retrover_output(context)

        return context


class MultiHopRetriever(adal.Retriever):
    def __init__(self, model_client, model_kwargs, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.passages_per_hop = passages_per_hop
        self.max_hops = max_hops

        self.data_parser = adal.DataClassParser(
            data_class=QueryRewritterData, return_data_class=True, format_type="json"
        )

        # Grad Component
        # self.query_generators: List[adal.Generator] = []
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
                        "few_shot_demos": Parameter(
                            name=f"few_shot_demos_{i}",
                            data=None,
                            role_desc="To provide few shot demos to the language model",
                            requires_opt=True,
                            param_type=ParameterType.DEMOS,
                        ),
                        "task_desc_str": Parameter(
                            name="task_desc_str",
                            data="""Write a simple search query that will help answer a complex question.

You will receive a context(may contain relevant facts) and a question.
Think step by step.""",
                            role_desc="Task description for the language model",
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
            self.deduplicaters.append(DeduplicateList())

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

    # def call(self, *, question: str, id: str = None) -> adal.RetrieverOutput:
    #     context = []
    #     print(f"question: {question}")
    #     for i in range(self.max_hops):
    #         gen_out = self.query_generators[i](
    #             prompt_kwargs={
    #                 "context": self.context_to_str(context),
    #                 "question": question,
    #             },
    #             id=id,
    #         )

    #         query = gen_out.data.query if gen_out.data and gen_out.data.query else None

    #         print(f"query {i}: {query}")

    #         retrieve_out = self.retrievers[i].call(input=query)
    #         passages = retrieve_out[0].documents
    #         context = self.deduplicate(context + passages)
    #     out = [adal.RetrieverOutput(documents=context, query=query, doc_indices=[])]
    #     return out

    # TODO: simplify and avoid the need where users need to write two methods (call and forward)
    def call(self, *, input: str, id: str = None) -> List[adal.RetrieverOutput]:
        # assemble the foundamental building blocks
        printc(f"question: {input}", "yellow")
        out = self.forward(input=input, id=id)

        if not isinstance(out, adal.Parameter):
            raise ValueError("The output should be a parameter")

        return out.data  # or full response its up to users

    def forward(self, *, input: str, id: str = None) -> adal.Parameter:
        # assemble the foundamental building blocks
        printc(f"question: {input}", "yellow")
        context = []

        queries: List[str] = []

        for i in range(self.max_hops):

            gen_out = self.query_generators[i].forward(
                prompt_kwargs={
                    "context": context,  # can be a list or a parameter
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
                x.full_response.data.query
                if x.full_response
                and x.full_response.data
                and x.full_response.data.query
                else (
                    x.full_response.raw_response
                    if x.full_response and x.full_response.raw_response
                    else None
                )
            )
            print(f"query {i}: {success_map_fn(gen_out)}")

            queries.append(success_map_fn(gen_out))

            gen_out.add_successor_map_fn(
                successor=self.retrievers[i], map_fn=success_map_fn
            )

            if success_map_fn(gen_out) is None:
                raise ValueError(f"The query is None, please check the generator {i}")

            retrieve_out = self.retrievers[i].forward(input=gen_out, id=id)

            def retrieve_out_map_fn(x: adal.Parameter):
                return x.data[0].documents if x.data and x.data[0].documents else []

            # print(f"retrieve_out: {retrieve_out}")

            retrieve_out.add_successor_map_fn(
                successor=self.deduplicaters[i], map_fn=retrieve_out_map_fn
            )

            context = self.deduplicaters[i].forward(
                exisiting_list=context, new_list=retrieve_out
            )

        context.param_type = ParameterType.RETRIEVER_OUTPUT

        def context_to_retrover_output(x):
            return [
                adal.RetrieverOutput(
                    documents=x.data, query=[input] + queries, doc_indices=[]
                )
            ]

        context.data = context_to_retrover_output(context)

        printc(f"MultiHopRetriever grad fn: {context.grad_fn}", "yellow")

        return context

    def backward(self, *args, **kwargs):

        printc(f"MultiHopRetriever backward: {args}", "yellow")
        super().backward(*args, **kwargs)
        return


from benchmarks.hotpot_qa.adal_exp.build_vanilla_rag import VanillaRAG


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


from adalflow.components.agent.react_v2 import ReActAgent


from benchmarks.hotpot_qa.adal_exp.build_vanilla_rag import (
    answer_template,
    AnswerData,
    task_desc_str,
)


# TODO: agent needs storage for the context instead of all in the step history.
class AgenticRAG(adal.GradComponent):
    def __init__(self, model_client, model_kwargs):
        super().__init__()

        self.dspy_retriever = DspyRetriever(top_k=3)
        self.llm_parser = adal.DataClassParser(
            data_class=AnswerData, return_data_class=True, format_type="json"
        )
        self.llm = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=answer_template,
            prompt_kwargs={
                "task_desc_str": adal.Parameter(
                    data=task_desc_str,
                    role_desc="Task description for the language model",
                    param_type=adal.ParameterType.PROMPT,
                    requires_opt=True,
                ),
                "few_shot_demos": adal.Parameter(
                    data=None,
                    requires_opt=None,
                    role_desc="To provide few shot demos to the language model",
                    param_type=adal.ParameterType.DEMOS,
                ),
                "output_format_str": self.llm_parser.get_output_format_str(),
            },
            output_processors=self.llm_parser,
        )

        self.context = []

        def dspy_retriever_as_tool(
            input: str,
            context_variables: Dict,
            id: Optional[str] = None,
        ) -> List[str]:
            r"""Retrieves the top k passages from using input as the query and save the documents in context_variables(Dict)'s context.

            Example: dspy_retriever_as_tool(subquery, context_variables=context_variables)
            Ensure you get all the context to answer the original question.
            """
            print(f"training: {self.dspy_retriever.training}")
            output = self.dspy_retriever(input=input, id=id)
            parsed_output = output
            if isinstance(output, adal.Parameter):
                parsed_output = output.data
                return output
            documents = parsed_output[0].documents
            if context_variables:
                context_variables["context"].extend(documents)
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
        #     print(f"context: {context}")
        #     output = self.llm(
        #         prompt_kwargs={"question": input, "context": context}, id=id
        #     )
        #     return output

        from adalflow.core.func_tool import FunctionTool

        tools = [
            FunctionTool(dspy_retriever_as_tool, component=self.dspy_retriever),
            # FunctionTool(generator_as_tool, component=self.llm),
        ]

        self.agent = ReActAgent(
            max_steps=3,
            add_llm_as_fallback=True,
            tools=tools,
            model_client=model_client,
            model_kwargs=model_kwargs,
            context_variables={"context": []},
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

    # eval mode
    output = multi_hop_retriever.call(question=question, id="1")
    print(output)

    # train mode
    multi_hop_retriever.train()
    output = multi_hop_retriever.forward(question=question, id="1")
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

    question = "How many storeys are in the castle that David Gregory inherited?"

    task.train()
    task(input=question, id="1")

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
