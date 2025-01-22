"""We will use dspy's retriever to keep that the same and only use our generator and optimizer"""

from typing import List, Optional, Union
from dataclasses import dataclass, field
import dspy

import adalflow as adal

from benchmarks.hotpot_qa.config import load_datasets

from adalflow.core.retriever import Retriever
from adalflow.core.types import RetrieverOutput
from adalflow.core import Generator


colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url="http://20.102.90.50:2017/wiki17_abstracts"
)

dspy.settings.configure(rm=colbertv2_wiki17_abstracts)


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


@dataclass
class AnswerData(adal.DataClass):
    reasoning: str = field(
        metadata={"desc": "The reasoning to produce the answer"},
    )
    answer: str = field(
        metadata={"desc": "The answer you produced"},
    )

    __output_fields__ = ["reasoning", "answer"]


query_template = """<START_OF_SYSTEM_PROMPT>
Write a simple search query that will help answer a complex question.

You will receive a context(may contain relevant facts) and a question.
Think step by step.

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

# Library gives a standard template for easy prompt
answer_template = """<START_OF_SYSTEM_PROMPT>
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


# Demonstrating how to wrap other retriever to adalflow retriever and be applied in training pipeline
# as a subclass of retriever which is a subclass of GradComponent, we dont need to do additional implementation
# data processing has already done


class DspyRetriever(Retriever):
    def __init__(self, top_k: int = 3):
        super().__init__()
        self.top_k = top_k
        self.dspy_retriever = dspy.Retrieve(k=top_k)

    def call(
        self, input: str, top_k: Optional[int] = None, id: str = None
    ) -> RetrieverOutput:

        k = top_k or self.top_k

        if not input:
            raise ValueError(f"Input cannot be empty, top_k: {k}")

        output = self.dspy_retriever(query_or_queries=input, k=k)
        # print(f"dsy_retriever output: {output}")
        documents = output.passages

        return RetrieverOutput(
            query=input,
            documents=documents,
            doc_indices=[],
        )


task_desc_str = r"""Answer questions with short factoid answers.

You will receive context(contain relevant facts).
Think step by step."""

task_desc_str_system_finetuned = "Generate a concise, factually accurate answer by synthesizing information from the provided context. If multiple sources are available, prioritize resolving ambiguities and cross-referencing data for consistency. Ensure the final answer directly addresses the question while considering specific numerical or descriptive criteria mentioned in the input."

# task_desc_str = r"""Answer questions with verbatim short factoid responses.

# You will receive context. Extract only the most relevant fact for a precise answer.
# """

demo_str = r"""reasoning: \"Dragon Data, the producer of Dragon 32/64, was based in Port Talbot, Wales,\\\n  \\ while TK82C was a product of a Brazilian company, Microdigital Eletr\\xF4nica Ltda.\"\nanswer: 'No'\n\nreasoning: The context specifies that the live action sequel '102 Dalmatians' was\n  directed by Kevin Lima.\nanswer: Kevin Lima\n\nreasoning: The context specifically mentions that in the 1970 Michigan gubernatorial\n  election, Republican William Milliken defeated Democrat Sander Levin.\nanswer: William Milliken\n\nreasoning: The context states that 'Lost Songs from the Lost Years' is a compilation\n  by Cloud Cult, which is an experimental indie rock band from Duluth, Minnesota.\nanswer: Minnesota
"""

# task_desc_str = r"""Answer the question with given context.
# The question requires you to answer one subquestion first, and then find the next potential subquestion and until you find the final answer.
# """


class VanillaRAG(adal.Component):
    def __init__(self, passages_per_hop=3, model_client=None, model_kwargs=None):
        super().__init__()

        self.passages_per_hop = passages_per_hop

        self.retriever = DspyRetriever(top_k=passages_per_hop)
        self.llm_parser = adal.DataClassParser(
            data_class=AnswerData, return_data_class=True, format_type="json"
        )
        self.llm = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs={
                "task_desc_str": adal.Parameter(
                    # data=task_desc_str_system_finetuned,
                    data=task_desc_str,
                    role_desc="""Task description for the language model,\
                    used with the following template: \
                    {{task_desc_str}} \
                    {{output_format_str}}\
                    <START_OF_USER>
Context: {{context}}
Question: {{question}}
<END_OF_USER>""",
                    param_type=adal.ParameterType.PROMPT,
                    requires_opt=True,
                    instruction_to_backward_engine="You need find the best way(where does the right answer come from the context) to extract the RIGHT answer from the context.",
                    instruction_to_optimizer="ou need find the best way(where does the right answer come from the context) to extract the RIGHT answer from the context.",
                    # + "Given existing context, ensure the task instructions can maximize the performance.",
                ),
                "few_shot_demos": adal.Parameter(
                    # data=demo_str,
                    data=None,
                    requires_opt=True,
                    role_desc="To provide few shot demos to the language model",
                    param_type=adal.ParameterType.DEMOS,
                ),
                "output_format_str": self.llm_parser.get_output_format_str(),
                # "output_format_str": adal.Parameter(
                #     data=self.llm_parser.get_output_format_str(),
                #     requires_opt=True,
                #     param_type=adal.ParameterType.PROMPT,
                #     role_desc="The output format string to ensure no failed json parsing",
                # ),
            },
            template=answer_template,
            output_processors=self.llm_parser,
            use_cache=True,
        )

    def call(self, question: str, id: str = None) -> adal.GeneratorOutput:
        if self.training:
            raise ValueError(
                "This component is not supposed to be called in training mode"
            )

        retriever_out = self.retriever.call(input=question, id=id)

        successor_map_fn = lambda x: (  # noqa E731
            "\n\n".join(x.documents) if x and x.documents else ""
        )
        retrieved_context = successor_map_fn(retriever_out)

        prompt_kwargs = {
            "context": retrieved_context,
            "question": question,
        }

        output = self.llm.call(
            prompt_kwargs=prompt_kwargs,
            id=id,
        )

        return output

    def forward(self, question: str, id: str = None) -> adal.Parameter:
        if not self.training:
            raise ValueError("This component is not supposed to be called in eval mode")
        retriever_out = self.retriever.forward(input=question, id=id)
        successor_map_fn = lambda x: (  # noqa E731
            "\n\n".join(x.data.documents)
            if x.data and x.data and x.data.documents
            else ""
        )
        retriever_out.add_successor_map_fn(successor=self.llm, map_fn=successor_map_fn)
        generator_out = self.llm.forward(
            prompt_kwargs={"question": question, "context": retriever_out}, id=id
        )
        return generator_out

    def bicall(
        self, question: str, id: str = None
    ) -> Union[adal.GeneratorOutput, adal.Parameter]:
        """You can also combine both the forward and call in the same function.
        Supports both training and eval mode by using __call__ for GradComponents
        like Retriever and Generator
        """
        retriever_out = self.retriever(input=question)
        if isinstance(retriever_out, adal.Parameter):
            successor_map_fn = lambda x: (  # noqa E731
                "\n\n".join(x.data.documents)
                if x.data and x.data and x.data.documents
                else ""
            )
            retriever_out.add_successor_map_fn(
                successor=self.llm, map_fn=successor_map_fn
            )
        else:
            successor_map_fn = lambda x: (  # noqa E731
                "\n\n".join(x.documents) if x and x.documents else ""
            )
            retrieved_context = successor_map_fn(retriever_out)
        prompt_kwargs = {
            "context": retrieved_context,
            "question": question,
        }
        output = self.llm(prompt_kwargs=prompt_kwargs, id=id)
        return output


class Vanilla(adal.Component):
    def __init__(self, passages_per_hop=3, model_client=None, model_kwargs=None):
        super().__init__()

        self.passages_per_hop = passages_per_hop

        # self.retriever = DspyRetriever(top_k=passages_per_hop)
        self.llm_parser = adal.DataClassParser(
            data_class=AnswerData, return_data_class=True, format_type="json"
        )
        self.llm = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs={
                "task_desc_str": adal.Parameter(
                    data=task_desc_str,
                    role_desc="Task description for the language model",
                    param_type=adal.ParameterType.PROMPT,
                    requires_opt=True,
                    instruction_to_backward_engine="You need find the best way(where does the right answer come from the context) to extract the RIGHT answer from the context.",
                    instruction_to_optimizer="You need find the best way(where does the right answer come from the context) to extract the RIGHT answer from the context.",
                    # + "Given existing context, ensure the task instructions can maximize the performance.",
                ),
                # "few_shot_demos": adal.Parameter(
                #     data=None,
                #     requires_opt=True,
                #     role_desc="To provide few shot demos to the language model",
                #     param_type=adal.ParameterType.DEMOS,
                # ),
                "output_format_str": self.llm_parser.get_output_format_str(),
            },
            template=answer_template,
            output_processors=self.llm_parser,
            use_cache=True,
        )

    def call(
        self, question: str, context: List[str], id: str = None
    ) -> adal.GeneratorOutput:
        if self.training:
            raise ValueError(
                "This component is not supposed to be called in training mode"
            )

        prompt_kwargs = {
            "context": context,
            "question": question,
        }

        output = self.llm.call(
            prompt_kwargs=prompt_kwargs,
            id=id,
        )

        return output

    # TODO: add id in the retriever output
    def forward(
        self, question: str, context: List[str], id: str = None
    ) -> adal.Parameter:
        if not self.training:
            raise ValueError("This component is not supposed to be called in eval mode")

        generator_out = self.llm.forward(
            prompt_kwargs={"question": question, "context": context}, id=id
        )
        return generator_out


def test_retriever():
    question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    retriever = DspyRetriever(top_k=3)
    retriever_out = retriever(input=question)
    print(f"retriever_out: {retriever_out}")


def test_vailla_rag():

    task = VanillaRAG(
        **gpt_3_model,
        passages_per_hop=3,
    )

    # test the retriever

    question = "How many storeys are in the castle that David Gregory inherited?"

    task.train()

    retriever_out = task.retriever(input=question)

    print(f"retriever_out: {retriever_out}")

    # test the forward function
    generator_out = task.forward(question=question, id="1")
    print(f"generator_out: {generator_out}")

    generator_out.draw_graph()

    # task.eval()
    # generator_out = task.call(question=question, id="1")
    # print(f"generator_out: {generator_out}")


from use_cases.config import (
    gpt_3_model,
)


def test_vanilla():
    task = Vanilla(
        **gpt_3_model,
        passages_per_hop=3,
    )
    task.eval()
    data_train, data_val, data_test = load_datasets()
    data = data_train[0]

    output = task.call(question=data.question, context=data.context, id="1")
    print(f"output: {output}, answer: {data.answer}")

    task.train()
    output = task.forward(question=data.question, context=data.context, id="1")
    print(f"output: {output.data}, answer: {data.answer}")


if __name__ == "__main__":
    # test_retriever()
    test_vanilla()
    # test_vailla_rag()
