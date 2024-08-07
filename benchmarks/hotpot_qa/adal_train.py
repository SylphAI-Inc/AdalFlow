"""We will use dspy's retriever to keep that the same and only use our generator and optimizer"""

import dspy
from lightrag.optim.parameter import Parameter, ParameterType

from lightrag.datasets.hotpot_qa import HotPotQA, HotPotQAData
from lightrag.datasets.types import Example


colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url="http://20.102.90.50:2017/wiki17_abstracts"
)

dspy.settings.configure(rm=colbertv2_wiki17_abstracts)


def load_datasets():
    # trainset = HotPotQA(split="train", size=2)
    # valset = HotPotQA(split="val", size=5)
    # testset = HotPotQA(split="test", size=5)
    trainset = HotPotQA(split="train", size=20)
    valset = HotPotQA(split="val", size=50)
    testset = HotPotQA(split="test", size=50)
    print(f"trainset, valset: {len(trainset)}, {len(valset)}, example: {trainset[0]}")
    return trainset, valset, testset


# task pipeline
from typing import Any, Tuple, Callable, Dict

from lightrag.core import Component, Generator


query_template = """<START_OF_SYSTEM_PROMPT>
Write a simple search query that will help answer a complex question.

You will receive a context and a question. Think step by step.
The last line of your response should be of the following format: 'Query: $VALUE' where VALUE is a search query.

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

You will receive context and a question. Think step by step.
The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a short factoid answer.

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

from lightrag.core.component import fun_to_component
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
    context: str = field(
        metadata={"desc": "The context to be used for answering the question"},
        default=None,
    )


class HotPotQARAG(Component):
    r"""Same system prompt as text-grad paper, but with our one message prompt template, which has better starting performance"""

    def __init__(self, model_client, model_kwargs, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.passages_per_hop = passages_per_hop
        self.max_hops = max_hops

        self.query_generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            # prompt_kwargs={
            #     "few_shot_demos": Parameter(
            #         alias="few_shot_demos",
            #         data=None,
            #         role_desc="To provide few shot demos to the language model",
            #         requires_opt=True,
            #         param_type=ParameterType.DEMOS,
            #     )
            # },
            template=query_template,
            output_processors=parse_string_query,
            use_cache=True,
        )
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        # TODO: sometimes the cache will collide, so we get different evaluation
        self.llm_counter = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs={
                "few_shot_demos": Parameter(
                    alias="few_shot_demos",
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
    def call(self, question: str, id: str = None) -> Any:  # Add id for tracing

        from dsp.utils import deduplicate

        context = []
        output = None
        for hop in range(self.max_hops):
            query = self.query_generator(
                prompt_kwargs={"context": context, "question": question}, id=id
            ).data
            # print(f"query: {query}")
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        # print(f"context: {context}")

        output = self.llm_counter(
            prompt_kwargs={"context": context, "question": question}, id=id
        )  # already support both training (forward + call)

        if not self.training:  # eval

            if output.data is None:
                error_msg = (
                    f"Error in processing the question: {question}, output: {output}"
                )
                print(error_msg)
                output = error_msg
            else:
                output = output.data

        return output


# Create adalcomponent

from lightrag.optim.trainer.adal import AdalComponent
from lightrag.optim.trainer.trainer import Trainer
from lightrag.optim.few_shot.bootstrap_optimizer import BootstrapFewShot
from lightrag.eval.answer_match_acc import AnswerMatchAcc
from lightrag.optim.text_grad.text_loss_with_eval_fn import EvalFnToTextLoss


class HotPotQARAGAdal(AdalComponent):
    # TODO: move teacher model or config in the base class so users dont feel customize too much
    def __init__(self, task: Component, teacher_model_config: dict):
        super().__init__()
        self.task = task
        self.teacher_model_config = teacher_model_config

        self.evaluator = AnswerMatchAcc("fuzzy_match")
        self.eval_fn = self.evaluator.compute_single_item

    def handle_one_train_sample(
        self, sample: HotPotQAData
    ) -> Any:  # TODO: auto id, with index in call train examples
        return self.task.call, {"question": sample.question, "id": sample.id}

    def handle_one_loss_sample(
        self, sample: HotPotQAData, y_pred: Any
    ) -> Tuple[Callable, Dict]:
        return self.loss_fn, {
            "kwargs": {
                "y": y_pred,
                "y_gt": Parameter(
                    data=sample.answer,
                    role_desc="The ground truth(reference correct answer)",
                    alias="y_gt",
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

    def evaluate_one_sample(self, sample: Any, y_pred: Any) -> Any:
        # print(f"sample: {sample}, y_pred: {y_pred}")
        return self.eval_fn(sample, y_pred)

    def configure_teacher_generator(self):
        super().configure_teacher_generator(**self.teacher_model_config)

    def configure_loss_fn(self):
        self.loss_fn = EvalFnToTextLoss(
            eval_fn=self.eval_fn,
            eval_fn_desc="ObjectCountingEvalFn, Output accuracy score: 1 for correct, 0 for incorrect",
            backward_engine=None,
        )


if __name__ == "__main__":
    ### Try the minimum effort to test on any task
    trainset, valset, testset = load_datasets()

    from use_cases.question_answering.bhh_object_count.config import gpt_3_model
    import dspy

    task = HotPotQARAG(
        **gpt_3_model,
        passages_per_hop=3,
        max_hops=2,
    )
    print(task)
    question = "How long is the highway Whitehorse/Cousins Airport was built to support as of 2012?"
    print(task(question))

    trainset, valset, testset = load_datasets()

    trainer = Trainer(
        adaltask=HotPotQARAGAdal(task=task, teacher_model_config=gpt_3_model),
        max_steps=4,
        raw_shots=0,
        bootstrap_shots=2,
        train_batch_size=4,
        ckpt_path="hotpot_qa_rag",
        strategy="random",
        save_traces=True,
        debug=True,  # make it having debug mode
    )
    # fit include max steps
    trainer.fit(
        train_dataset=trainset, val_dataset=valset, test_dataset=testset, debug=False
    )


# TODO: i forgot that i need demo_data_class
# TODO: i forgot that i need to set id
# Failed to generate demos but no error messages
