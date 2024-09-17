"""We will use dspy's retriever to keep that the same and only use our generator and optimizer"""

from typing import List, Optional
from dataclasses import dataclass, field
import dspy

import adalflow as adal

from adalflow.datasets.hotpot_qa import HotPotQA

from adalflow.core.retriever import Retriever
from adalflow.core.types import RetrieverOutput
from adalflow.core import Generator


colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url="http://20.102.90.50:2017/wiki17_abstracts"
)

dspy.settings.configure(rm=colbertv2_wiki17_abstracts)


def load_datasets():

    trainset = HotPotQA(split="train", size=20)
    valset = HotPotQA(split="val", size=50)
    testset = HotPotQA(split="test", size=50)
    print(f"trainset, valset: {len(trainset)}, {len(valset)}, example: {trainset[0]}")
    return trainset, valset, testset


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

    def call(self, input: str, top_k: Optional[int] = None) -> List[RetrieverOutput]:

        k = top_k or self.top_k

        output = self.dspy_retriever(query_or_queries=input, k=k)
        # print(f"dsy_retriever output: {output}")
        final_output: List[RetrieverOutput] = []
        documents = output.passages

        final_output.append(
            RetrieverOutput(
                query=input,
                documents=documents,
                doc_indices=[],
            )
        )
        # print(f"final_output: {final_output}")
        return final_output


task_desc_str = r"""Answer questions with short factoid answers.

You will receive context(may contain relevant facts) and a question.
Think step by step."""


class VanillaRAG(adal.GradComponent):
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
                    data=task_desc_str,
                    role_desc="Task description for the language model",
                    param_type=adal.ParameterType.PROMPT,
                ),
                "few_shot_demos": adal.Parameter(
                    data=None,
                    requires_opt=True,
                    role_desc="To provide few shot demos to the language model",
                    param_type=adal.ParameterType.DEMOS,
                ),
                "output_format_str": self.llm_parser.get_output_format_str(),
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
        # user should just treat it as a call function
        # and we will handle the connection between the components
        # they should directly pass the retriever_output along with
        # each output's successor_map_fn.
        # what if it is passed to two different componnents?
        # we can create a copy
        retriever_out = self.retriever.call(input=question)

        successor_map_fn = lambda x: (  # noqa E731
            "\n\n".join(x[0].documents) if x and x[0] and x[0].documents else ""
        )
        retrieved_context = successor_map_fn(retriever_out)

        # print(f"retrieved_context: {retrieved_context}")
        # print(f"retriever_out: {retriever_out}")
        prompt_kwargs = {
            "context": retrieved_context,
            "question": question,
        }

        output = self.llm.call(
            prompt_kwargs=prompt_kwargs,
            id=id,
        )
        # self.llm.print_prompt(**prompt_kwargs)
        return output

    def forward(self, question: str, id: str = None) -> adal.Parameter:
        if not self.training:
            raise ValueError("This component is not supposed to be called in eval mode")
        # TODO: add id in the retriever output
        retriever_out = self.retriever.forward(input=question)
        successor_map_fn = lambda x: (  # noqa E731
            "\n\n".join(x.data[0].documents)
            if x.data and x.data[0] and x.data[0].documents
            else ""
        )
        retriever_out.add_successor_map_fn(successor=self.llm, map_fn=successor_map_fn)
        generator_out = self.llm.forward(
            prompt_kwargs={"question": question, "context": retriever_out}, id=id
        )
        return generator_out


def test_vailla_rag():

    from use_cases.config import (
        gpt_3_model,
    )

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

    task.eval()
    generator_out = task.call(question=question, id="1")
    print(f"generator_out: {generator_out}")


if __name__ == "__main__":
    test_vailla_rag()
