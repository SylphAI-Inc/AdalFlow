"""Implementation of G-Eval: G-eval <https://arxiv.org/abs/2303.08774, https://github.com/nlpyang/geval>
Instead of getting 1/5 as the score, AdalFlow will use 0.2 as the score, so that we can have a score in range [0, 1] for all metrics."""

from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import logging


from adalflow.eval.base import BaseEvaluator
from adalflow.core.component import Component
from adalflow.core.model_client import ModelClient
from adalflow.eval.llm_as_judge import DEFAULT_LLM_EVALUATOR_MODEL_KWARGS
from adalflow.core.string_parser import FloatParser

__all__ = ["GEvalMetric", "NLGTask", "GEvalLLMJudge", "GEvalJudgeEvaluator"]

log = logging.getLogger(__name__)


class GEvalMetric(Enum):

    RELEVANCE = "Relevance"  # range [1, 5]
    FLUENCY = "Fluency"  # range [1, 3]
    CONSISTENCY = "Consistency"  # range [1, 5]
    COHERENCE = "Coherence"  # range [1, 5]


all_geval_metrics = [m.value for m in GEvalMetric]
all_geval_metrics_to_range = {
    "Relevance": 5.0,
    "Fluency": 3.0,
    "Consistency": 5.0,
    "Coherence": 5.0,
}


class NLGTask(Enum):
    SUMMARIZATION = {
        "task_desc_str": r"""You will be given a summary of a text.  Please evaluate the summary based on the following criteria:""",
        "criteria_relevance": r"""Relevance (1-5) - selection of important content from the source.
        The summary should include only important information from the source document.
        Annotators were instructed to penalize summaries which contained redundancies and excess information.""",
        "steps_relevance": r"""1. Read the summary and the source document carefully.
        2. Compare the summary to the source document and identify the main points of the article.
        3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
        4. Assign a relevance score from 1 to 5.""",
        "criteria_fluency": r"""Fluency (1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.
        - 1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
        - 2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
        - 3: Good. The summary has few or no errors and is easy to read and follow.
        """,
        "steps_fluency": None,
        "criteria_coherence": r"""Coherence (1-5) - the collective quality of all sentences.
        We align this dimension with the DUC quality question of structure and coherence whereby "the summary should be well-structured and well-organized.
        The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic.""",
        "steps_coherence": r"""1. Read the input text carefully and identify the main topic and key points.
        2. Read the summary and assess how well it captures the main topic and key points. And if it presents them in a clear and logical order.
        3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.""",
        "criteria_consistency": r"""Consistency (1-5) - the factual alignment between the summary and the summarized source.
        A factually consistent summary contains only statements that are entailed by the source document.
        Annotators were also asked to penalize summaries that contained hallucinated facts. """,
        "steps_consistency": r"""1. Read the summary and the source document carefully.
        2. Identify the main facts and details it presents.
        3. Read the summary and compare it to the source document to identify any inconsistencies or factual errors that are not supported by the source.
        4. Assign a score for consistency based on the Evaluation Criteria.""",
    }


DEFAULT_G_EVAL_RPROMPT = r"""
<START_OF_SYSTEM_PROMPT>
{# task desc #}
{{task_desc_str}}
---------------------
{# evaluation criteria #}
Evaluation Criteria:
{{evaluation_criteria_str}}
---------------------
{# evaluation steps #}
{% if evaluation_steps_str %}
Evaluation Steps:
{{evaluation_steps_str}}
---------------------
{% endif %}
{{input_str}}
{ # evaluation form #}
Output the score for metric (scores ONLY!): {{metric_name}}
<END_OF_SYSTEM_PROMPT>
"""


class GEvalLLMJudge(Component):
    __doc__ = r"""Demonstrate how to use an LLM/Generator to output True or False for a judgement query.

    You can use any of your template to adapt to more tasks and sometimes you can directly ask LLM to output a score in range [0, 1] instead of only True or False.

    A call on the LLM judge equalize to _compute_single_item method.

    Args:
        model_client (ModelClient): The model client to use for the generator.
        model_kwargs (Dict[str, Any], optional): The model kwargs to pass to the model client. Defaults to {}. Please refer to :ref:`ModelClient<components-model_client>` for the details on how to set the model_kwargs for your specific model if it is from our library.
        template (str, optional): The template to use for the LLM evaluator. Defaults to None.
        use_cache (bool, optional): Whether to use cache for the LLM evaluator. Defaults to True.
        default_task (NLGTask, optional): The default task to use for the judgement query. Defaults to None.
   """

    def __init__(
        self,
        model_client: Optional[ModelClient] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        template: Optional[str] = None,
        use_cache: bool = True,
        default_task: Optional[NLGTask] = None,
    ):
        from adalflow.core.generator import Generator

        super().__init__()
        self.model_client = model_client
        if model_client is None:
            log.info("model_client is None, default to OpenAIClient.")
            try:
                from adalflow.components.model_client import OpenAIClient
            except ImportError:
                raise ImportError(
                    "OpenAIClient is not available. Please fix the import error or set your own choice of model_client and model_kwargs."
                )
            self.model_client = OpenAIClient()
        self.model_kwargs = model_kwargs or DEFAULT_LLM_EVALUATOR_MODEL_KWARGS
        self.template = template or DEFAULT_G_EVAL_RPROMPT
        self.prompt_kwargs = {k: {} for k in all_geval_metrics}
        self.default_task = default_task
        if default_task:
            # task_name = default_task.name
            # print(f"task_name: {task_name}")
            for metric_name in all_geval_metrics:
                metric_name_lower = metric_name.lower()
                self.prompt_kwargs[metric_name] = {
                    "task_desc_str": default_task.value["task_desc_str"],
                    "evaluation_criteria_str": default_task.value[
                        f"criteria_{metric_name_lower}"
                    ],
                    "evaluation_steps_str": default_task.value.get(
                        f"steps_{metric_name_lower}"
                    ),
                    "metric_name": metric_name,
                }
        self.llm_evaluator = Generator(
            model_client=self.model_client,
            model_kwargs=self.model_kwargs,
            template=self.template,
            use_cache=use_cache,
            output_processors=FloatParser(),
        )

    def call(self, input_str: str) -> Dict[str, Any]:
        r"""
        Pass the input string with all information to the LLM evaluator and get the judgement.

        Args:
            input_str (str): The input string with all information.

        Returns:
            Dict[str, Any]: The judgement result.
        """

        output = {}

        total_scores = 0
        num_metrics = 0

        for metric_name in all_geval_metrics:
            # add input_str to prompt_kwargs
            self.prompt_kwargs[metric_name]["input_str"] = input_str

            metric_score = self.llm_evaluator(
                prompt_kwargs=self.prompt_kwargs[metric_name]
            )
            # print(
            #     f"prompt: \n{self.llm_evaluator.get_prompt(**self.prompt_kwargs[metric_name])}"
            # )
            output[metric_name] = (
                metric_score.data / all_geval_metrics_to_range[metric_name]
                if metric_score and metric_score.data
                else None
            )
            # print(f"metric score: {metric_score}")
            if output[metric_name]:
                total_scores += output[metric_name]
                num_metrics += 1
        output["overall"] = total_scores / num_metrics if num_metrics else None

        return output

    def _extra_repr(self) -> str:
        s = f"default_task= {self.default_task}, prompt_kwargs={self.prompt_kwargs}"
        return s


class GEvalJudgeEvaluator(BaseEvaluator):
    r"""
    LLM as judge for evaluating the performance of a LLM in form of GEval with 4 main metrics:

    Relevance, Fluency, Consistency, Coherence.

    Args:
        llm_judge (Component, optional): The LLM evaluator to use. Defaults to GEvalLLMJudge().
    """

    def __init__(
        self,
        llm_judge: Optional[Component] = None,
    ):
        super().__init__()
        self.llm_judge = llm_judge or GEvalLLMJudge()

    def compute_single_item(self, input_str: str) -> Dict[str, Any]:
        r"""
        Compute the score for a single item.

        Args:
            input_str (str): The input string with all information.

        Returns:
            Dict[str, Any]: The judgement result.
        """
        return self.llm_judge(input_str=input_str)

    def compute(
        self,
        input_strs: List[str],
    ) -> Tuple[Dict, List[Dict[str, Any]]]:
        r"""
        Get the judgement of the predicted answer for a list of questions.

        Args:
           input_strs (List[str]): List of input strings.
        Returns:
            List[Dict[str, Any]]: The judgement result.
        """
        output = []
        for input_str in input_strs:
            output.append(self.compute_single_item(input_str))

        # average across different keys
        final_output = {metric: [] for metric in all_geval_metrics}
        final_output.update({"overall": []})
        for data in output:
            for metric, score in data.items():
                if score is None:
                    continue
                final_output[metric].append(score)

        for metric, scores in final_output.items():
            if not scores:
                final_output[metric] = None
            else:
                final_output[metric] = sum(scores) / len(scores)

        return final_output, output

    def __str__(self) -> str:
        s = f"llm_judge={self.llm_judge}, prompt_kwargs={self.llm_judge.prompt_kwargs}"
        return s
