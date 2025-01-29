"deprecated"
"""We will use dspy's retriever to keep that the same and only use our generator and optimizer"""

import dspy
from typing import List, Union, Optional, Dict, Callable
from dataclasses import dataclass, field

import adalflow as adal
from adalflow.optim.parameter import Parameter, ParameterType

from adalflow.datasets.hotpot_qa import HotPotQA, HotPotQAData
from adalflow.datasets.types import Example

from adalflow.core.retriever import Retriever


colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url="http://20.102.90.50:2017/wiki17_abstracts"
)

dspy.settings.configure(rm=colbertv2_wiki17_abstracts)


def load_datasets():

    trainset = HotPotQA(split="train", size=20)  # 20
    valset = HotPotQA(split="val", size=50)  # 50
    testset = HotPotQA(split="test", size=50)  # to keep the same as the dspy #50
    print(f"trainset, valset: {len(trainset)}, {len(valset)}, example: {trainset[0]}")
    return trainset, valset, testset


# task pipeline
from typing import Any, Tuple

from adalflow.core import Component, Generator


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
Answer questions with short factoid answers.

You will receive context(may contain relevabt facts) and a question.
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
"""

from adalflow.core.component import fun_to_component
import re


@fun_to_component
def parse_string_query(text: str) -> str:
    return re.search(r"Query: (.*)", text).group(1)


@fun_to_component
def parse_string_answer(text: str) -> str:
    return re.search(r"Answer: (.*)", text).group(1)


from dataclasses import dataclass, field


@dataclass
class HotPotQADemoData(Example):
    context: List[str] = field(
        metadata={"desc": "The context to be used for answering the question"},
        default_factory=list,
    )
    score: float = field(
        metadata={"desc": "The score of the answer"},
        default=None,
    )


from benchmarks.hotpot_qa.dspy_train import validate_context_and_answer_and_hops


def convert_y_pred_to_dataclass(y_pred):
    # y_pred in both eval and train mode
    context: List[str] = (
        y_pred.input_args["prompt_kwargs"]["context"]
        if hasattr(y_pred, "input_args")
        else []
    )
    # context_str = "\n".join(context)
    data = y_pred.data if hasattr(y_pred, "data") else y_pred
    return DynamicDataClassFactory.from_dict(
        class_name="HotPotQAData",
        data={
            "answer": data,
            "context": context,
        },
    )


def eval_fn(sample, y_pred, metadata):
    if isinstance(sample, Parameter):
        sample = sample.data
    y_pred_obj = convert_y_pred_to_dataclass(y_pred)
    return 1 if validate_context_and_answer_and_hops(sample, y_pred_obj) else 0


from adalflow.core.types import RetrieverOutput, GeneratorOutput


# Demonstrating how to wrap other retriever to adalflow retriever and be applied in training pipeline
class DspyRetriever(Retriever):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.dspy_retriever = dspy.Retrieve(k=k)

    def call(self, input: str) -> List[RetrieverOutput]:
        output = self.dspy_retriever(query_or_queries=input, k=self.k)
        print(f"dsy_retriever output: {output}")
        final_output: List[RetrieverOutput] = []
        documents = output.passages

        final_output.append(
            RetrieverOutput(
                query=input,
                documents=documents,
                doc_indices=[],
            )
        )
        print(f"final_output: {final_output}")
        return final_output


# example need to have question,
# pred needs to have query

import adalflow as adal


# User customize an auto-grad operator
class MultiHopRetriever(adal.Retriever):
    def __init__(self, model_client, model_kwargs, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.passages_per_hop = passages_per_hop
        self.max_hops = max_hops

        self.data_parser = adal.DataClassParser(
            data_class=QueryRewritterData, return_data_class=True, format_type="yaml"
        )

        # Grad Component
        self.query_generator = Generator(
            name="query_generator",
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs={
                "few_shot_demos": Parameter(
                    name="few_shot_demos_1",
                    data=None,
                    role_desc="To provide few shot demos to the language model",
                    requires_opt=True,
                    param_type=ParameterType.DEMOS,
                ),
                "output_format_str": self.data_parser.get_output_format_str(),
            },
            template=query_template,
            # output_processors=parse_string_query,
            output_processors=self.data_parser,
            use_cache=True,
            # demo_data_class=HotPotQADemoData,
            # demo_data_class_input_mapping={
            #     "question": "question",
            #     # "context": "context",
            # },
            # demo_data_class_output_mapping={"answer": lambda x: x.raw_response},
        )
        self.retrieve = DspyRetriever(k=passages_per_hop)

    @staticmethod
    def context_to_str(context: List[str]) -> str:
        return "\n".join(context)

    def call(self, *, question: str, id: str = None) -> Any:  # Add id for tracing
        # inference mode!!!
        # output = self.forward(question, id=id)

        context = []
        self.max_hops = 1
        for hop in range(self.max_hops):
            gen_out = self.query_generator(
                prompt_kwargs={
                    "context": self.context_to_str(context),
                    "question": question,
                },
                id=id,
            )
            query = None
            # TODO: the bridge between the retriever to the generator and generator to the retriever needs to be more smooth
            if isinstance(gen_out, GeneratorOutput):
                query = (  # noqa: F841
                    gen_out.data.query if gen_out.data and gen_out.data.query else None
                )
            elif isinstance(gen_out, adal.Parameter):
                gen_out.successor_map_fn = lambda x: (
                    x.full_response.data.query
                    if x.full_response and x.full_response.data
                    else None
                )
                print(f"gen_out: {gen_out}")
                # query = (
                #     gen_out.full_response.data.query
                #     if gen_out.full_response and gen_out.full_response.data
                #     else None
                # )
            retrieve_out = self.retrieve(input=gen_out)
            print(f"retrieve_out: {retrieve_out}")
            # passages = []
            # if isinstance(retrieve_out, Parameter):
            #     passages = retrieve_out.data[0].documents
            # else:
            #     passages = retrieve_out[0].documents

            # print(f"passages: {passages}")

            # context = deduplicate(context + passages)

        # # for hop in range(self.max_hops):
        # last_context_param = Parameter(
        #     data=context,
        #     name=f"query_context_{id}_{0}",
        #     requires_opt=True,
        # )
        # query = self.query_generator(
        #     prompt_kwargs={
        #         "context": last_context_param,
        #         "question": question,
        #     },
        #     id=id,
        # )
        # print(f"query: {query}")
        # if isinstance(query, GeneratorOutput):
        #     query = query.data
        # output = self.retrieve(query)
        # print(f"output: {output}")
        # print(f"output call: {output}")
        # return output[0].documents

    # def forward(self, question: str, id: str = None) -> Parameter:
    #     question_param = question
    #     if not isinstance(question, Parameter):
    #         question_param = Parameter(
    #             data=question,
    #             name="question",
    #             role_desc="The question to be answered",
    #             requires_opt=False,
    #         )
    #     context = []
    #     self.max_hops = 1
    #     # for hop in range(self.max_hops):
    #     last_context_param = Parameter(
    #         data=context,
    #         name=f"query_context_{id}_{0}",
    #         requires_opt=True,
    #     )
    #     query = self.query_generator(
    #         prompt_kwargs={
    #             "context": last_context_param,
    #             "question": question_param,
    #         },
    #         id=id,
    #     )
    #     print(f"query: {query}")
    #     if isinstance(query, GeneratorOutput):
    #         query = query.data
    #     output = self.retrieve(query)
    #     print(f"output: {output}")
    #     passages = []
    #     if isinstance(output, Parameter):
    #         passages = output.data[0].documents
    #     else:
    #         passages = output[0].documents
    #     # context = deduplicate(context + passages) # all these needs to gradable
    #     # output_param = Parameter(
    #     #     data=passages,
    #     #     alias=f"qa_context_{id}",
    #     #     role_desc="The context to be used for answering the question",
    #     #     requires_opt=True,
    #     # )
    #     output.data = passages  # reset the values to be used in the next
    #     if not isinstance(output, Parameter):
    #         raise ValueError(f"Output must be a Parameter, got {output}")
    #     return output
    #     # output_param.set_grad_fn(
    #     #     BackwardContext(
    #     #         backward_fn=self.backward,
    #     #         response=output_param,
    #     #         id=id,
    #     #         prededecessors=prededecessors,
    #     #     )
    #     # )
    #     # return output_param

    def backward(self, response: Parameter, id: Optional[str] = None):
        print(f"MultiHopRetriever backward: {response}")
        children_params = response.predecessors
        # backward score to the demo parameter
        for pred in children_params:
            if pred.requires_opt:
                # pred._score = float(response._score)
                pred.set_score(response._score)
                print(
                    f"backpropagate the score {response._score} to {pred.name}, is_teacher: {self.teacher_mode}"
                )
                if pred.param_type == ParameterType.DEMOS:
                    # Accumulate the score to the demo
                    pred.add_score_to_trace(
                        trace_id=id, score=response._score, is_teacher=self.teacher_mode
                    )
                    print(f"Pred: {pred.name}, traces: {pred._traces}")


class HotPotQARAG(
    Component
):  # use component as not creating a new ops, but assemble existing ops
    r"""Same system prompt as text-grad paper, but with our one message prompt template, which has better starting performance"""

    def __init__(self, model_client, model_kwargs, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.passages_per_hop = passages_per_hop
        self.max_hops = max_hops

        self.multi_hop_retriever = MultiHopRetriever(
            model_client=model_client,
            model_kwargs=model_kwargs,
            passages_per_hop=passages_per_hop,
            max_hops=max_hops,
        )
        # TODO: sometimes the cache will collide, so we get different evaluation
        self.llm_counter = Generator(
            name="QuestionAnswering",
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs={
                "few_shot_demos": Parameter(
                    name="few_shot_demos",
                    data=None,
                    role_desc="To provide few shot demos to the language model",
                    requires_opt=True,
                    param_type=ParameterType.DEMOS,
                )
            },
            template=answer_template,
            output_processors=parse_string_answer,
            use_cache=True,
            demo_data_class=HotPotQADemoData,
            demo_data_class_input_mapping={
                "question": "question",
                "context": "context",
            },
            demo_data_class_output_mapping={"answer": lambda x: x.raw_response},
        )

    # TODO: the error will be a context
    # a component wont handle training, forward or backward, just passing everything through
    def call(self, question: str, id: str = None) -> Union[Parameter, str]:

        # normal component, will be called when in inference mode

        question_param = Parameter(
            data=question,
            name="question",
            role_desc="The question to be answered",
            requires_opt=False,
        )
        context = []  # noqa: F841
        output = None
        retrieved_context = self.multi_hop_retriever(question_param, id=id)

        # forming a backpropagation graph
        # Make this step traceable too.
        # for hop in range(self.max_hops):
        #     # make context a parameter to be able to trace
        #     query = self.query_generator(
        #         prompt_kwargs={
        #             "context": Parameter(
        #                 data=context, alias=f"query_context_{id}", requires_opt=True
        #             ),
        #             "question": question_param,
        #         },
        #         id=id,
        #     )
        #     print(f"query: {query}")
        #     if isinstance(query, GeneratorOutput):
        #         query = query.data
        #     output = self.retrieve(query)
        #     print(f"output: {output}")
        #     passages = []
        #     if isinstance(output, Parameter):
        #         passages = output.data[0].documents
        #     else:
        #         output[0].documents
        #     context = deduplicate(context + passages)
        # print(f"context: {context}")

        output = self.llm_counter(
            prompt_kwargs={
                "context": retrieved_context,
                "question": question_param,
            },
            id=id,
        )  # already support both training (forward + call)

        if (
            not self.training
        ):  # if users want to customize the output, ensure to use if not self.training

            # convert the generator output to a normal data format
            print(f"converting output: {output}")

            if output.data is None:
                error_msg = (
                    f"Error in processing the question: {question}, output: {output}"
                )
                print(error_msg)
                output = error_msg
            else:
                output = output.data
        return output


from adalflow.optim.trainer.adal import AdalComponent
from adalflow.optim.trainer.trainer import Trainer
from adalflow.optim.few_shot.bootstrap_optimizer import BootstrapFewShot
from adalflow.eval.answer_match_acc import AnswerMatchAcc
from adalflow.optim.text_grad.text_loss_with_eval_fn import EvalFnToTextLoss
from adalflow.core.base_data_class import DynamicDataClassFactory


class HotPotQARAGAdal(AdalComponent):
    # TODO: move teacher model or config in the base class so users dont feel customize too much
    def __init__(self, task: Component, teacher_model_config: dict):
        super().__init__()
        self.task = task
        self.teacher_model_config = teacher_model_config

        self.evaluator = AnswerMatchAcc("fuzzy_match")
        self.eval_fn = self.evaluator.compute_single_item
        # self.eval_fn = eval_fn

    def handle_one_task_sample(
        self, sample: HotPotQAData
    ) -> Any:  # TODO: auto id, with index in call train examples
        return self.task, {"question": sample.question, "id": sample.id}

    def handle_one_loss_sample(
        self, sample: HotPotQAData, y_pred: Any
    ) -> Tuple[Callable, Dict]:
        return self.loss_fn.forward, {
            "kwargs": {
                "y": y_pred,
                "y_gt": Parameter(
                    data=sample.answer,
                    role_desc="The ground truth(reference correct answer)",
                    name="y_gt",
                    requires_opt=False,
                ),
            }
        }

    def configure_optimizers(self, *args, **kwargs):

        # TODO: simplify this, make it accept generator
        parameters = []
        for name, param in self.task.named_parameters():
            param.name = name
            parameters.append(param)
        do = BootstrapFewShot(params=parameters)
        return [do]

    def evaluate_one_sample(
        self, sample: Any, y_pred: Any, metadata: Dict[str, Any]
    ) -> Any:

        # we need "context" be passed as metadata
        # print(f"sample: {sample}, y_pred: {y_pred}")
        # convert pred to Dspy structure

        # y_obj = convert_y_pred_to_dataclass(y_pred)
        # print(f"y_obj: {y_obj}")
        # raise ValueError("Stop here")
        if metadata:
            return self.eval_fn(sample, y_pred, metadata)
        return self.eval_fn(sample, y_pred)

    def configure_teacher_generator(self):
        super().configure_teacher_generator(**self.teacher_model_config)

    def configure_loss_fn(self):
        self.loss_fn = EvalFnToTextLoss(
            eval_fn=self.eval_fn,
            eval_fn_desc="ObjectCountingEvalFn, Output accuracy score: 1 for correct, 0 for incorrect",
            backward_engine=None,
        )


def validate_dspy_demos(
    demos_file="benchmarks/BHH_object_count/models/dspy/hotpotqa.json",
):
    from adalflow.utils.file_io import load_json

    demos_json = load_json(demos_file)

    demos = demos_json["generate_answer"]["demos"]  # noqa: F841

    # task = HotPotQARAG(  # noqa: F841
    #     **gpt_3_model,
    #     passages_per_hop=3,
    #     max_hops=2,
    # )
    # task.llm_counter.p


def test_multi_hop_retriever():

    from use_cases.config import (
        gpt_3_model,
    )

    multi_hop_retriever = MultiHopRetriever(
        **gpt_3_model,
        passages_per_hop=3,
        max_hops=2,
    )
    # 1. use print
    # print(multi_hop_retriever.query_generator)
    # # 2. run one forward for query generator
    question = "How many storeys are in the castle that David Gregory inherited?"
    # context = []
    # context_str = multi_hop_retriever.context_to_str(context)
    # print(
    #     multi_hop_retriever.query_generator(
    #         prompt_kwargs={"question": question, "context": context_str}, id="1"
    #     )
    # )
    # # verfify the prompt
    # multi_hop_retriever.query_generator.print_prompt(
    #     **{"question": question, "context": context_str}
    # )

    # training mode
    multi_hop_retriever.train()

    # 3. run one forward for retriever
    print(multi_hop_retriever(question=question, id="1"))


def train():
    trainset, valset, testset = load_datasets()

    from use_cases.config import (
        gpt_3_model,
        gpt_4o_model,
    )

    task = HotPotQARAG(
        **gpt_3_model,
        passages_per_hop=3,
        max_hops=2,
    )
    print(task)
    question = "How long is the highway Whitehorse/Cousins Airport was built to support as of 2012?"
    print(task(question))

    # for name, param in task.named_parameters():
    #     print(f"name: {name}, param: {param}")

    trainset, valset, testset = load_datasets()

    trainer = Trainer(
        adaltask=HotPotQARAGAdal(task=task, teacher_model_config=gpt_4o_model),
        max_steps=10,
        raw_shots=0,
        bootstrap_shots=4,
        train_batch_size=4,
        ckpt_path="hotpot_qa_rag",
        strategy="random",
        save_traces=True,
        debug=True,  # make it having debug mode
        weighted_sampling=True,
    )
    # fit include max steps
    trainer.fit(
        train_dataset=trainset, val_dataset=valset, test_dataset=testset, debug=True
    )


if __name__ == "__main__":
    ### Try the minimum effort to test on any task

    # get_logger(level="DEBUG")
    test_multi_hop_retriever()


# TODO: i forgot that i need demo_data_class
# TODO: i forgot that i need to set id
# Failed to generate demos but no error messages
