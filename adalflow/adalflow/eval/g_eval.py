"""Implementation of G-Eval: G-eval <https://arxiv.org/abs/2303.08774, https://github.com/nlpyang/geval>"""

from enum import Enum
from typing import Dict, Any, Optional, List
import logging


from adalflow.eval.base import BaseEvaluator
from adalflow.core.component import Component
from adalflow.core.model_client import ModelClient
from adalflow.eval.llm_as_judge import DEFAULT_LLM_EVALUATOR_MODEL_KWARGS
from adalflow.core.string_parser import FloatParser

log = logging.getLogger(__name__)


class GEvalMetric(Enum):

    RELEVANCE = "Relevance"  # range [1, 5]
    FLUENCY = "Fluency"  # range [1, 3]
    CONSISTENCY = "Consistency"  # range [1, 5]
    COHERENCE = "Coherence"  # range [1, 5]


all_geval_metrics = [m.value for m in GEvalMetric]


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
Evaluation Form (scores ONLY):
- {{metric_name}}:

Output the score only.
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
        jugement_query (str, optional): The judgement query string. Defaults to DEFAULT_JUDGEMENT_QUERY.
        output_type (Literal["bool", "float"], optional): The output type of the judgement. Defaults to "bool".
        use_cache (bool, optional): Whether to use cache for the LLM evaluator. Defaults to True.

    Note:
        Must use True/False instead of Yes/No in the judgement_query for response.
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
        self.prompt_kwargs = {k: {} for k in GEvalMetric}
        self.default_task = default_task
        if default_task:
            task_name = default_task.name
            print(f"task_name: {task_name}")
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
            print(
                f"prompt: {self.llm_evaluator.get_prompt(**self.prompt_kwargs[metric_name])}"
            )
            output[metric_name] = (
                metric_score.data if metric_score and metric_score.data else None
            )
            if output[metric_name]:
                total_scores += output[metric_name]
                num_metrics += 1
        output["overall"] = total_scores / num_metrics if num_metrics else None

        return output

    def _extra_repr(self) -> str:
        s = f"default_task= {self.default_task}, "
        return s


class GEvalJudgeEvaluator(BaseEvaluator):
    r"""
    LLM as judge for evaluating the performance of a LLM.

    Args:
        llm_evaluator (Component, optional): The LLM evaluator to use. Defaults to DefaultLLMJudge.

    Examples:
        >>> questions = [
        "Is Beijing in China?",
        "Is Apple founded before Google?",
        "Is earth flat?",
        ]
        >>> pred_answers = ["Yes", "Yes, Appled is founded before Google", "Yes"]
        >>> gt_answers = ["Yes", "Yes", "No"]
        >>> judgement_query = "For the question, does the predicted answer contain the ground truth answer?"
        >>> llm_judge = LLMasJudge()
        >>> avg_judgement, judgement_list = llm_judge.compute(
        questions, gt_answers, pred_answers, judgement_query
        )
        >>> avg_judgement
        2 / 3
        >>> judgement_list
        [True, True, False]

    Customize the LLMJudge

    .. code-block:: python

        llm_judge = Def
    """

    def __init__(
        self,
        llm_evaluator: Optional[Component] = None,
    ):
        super().__init__()
        self.llm_evaluator = llm_evaluator or GEvalLLMJudge()

    def compute(
        self,
        input_strs: List[str],
    ) -> List[Dict[str, Any]]:
        r"""
        Get the judgement of the predicted answer for a list of questions.

        Args:
           input_strs (List[str]): List of input strings.
        Returns:
            List[Dict[str, Any]]: The judgement result.
        """
        output = []
        for input_str in input_strs:
            output.append(self.llm_evaluator(input_str=input_str))
        return output

    def __str__(self) -> str:
        s = f"llm_evaluator={self.llm_evaluator}"
        return s


if __name__ == "__main__":
    from adalflow.utils import setup_env

    setup_env()

    model_kwargs = {
        "model": "gpt-4o",
        "n": 20,
        "top_p": 1,
        "max_tokens": 5,
        "temperature": 1,
    }

    g_eval = GEvalLLMJudge(
        default_task=NLGTask.SUMMARIZATION, model_kwargs=model_kwargs
    )
    input_template = """Source Document: {source}
    Summary: {summary}
    """

    input_str = input_template.format(
        source="Paul Merson has restarted his row with Andros Townsend after the Tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with Burnley on Sunday . 'Just been watching the game , did you miss the coach ? # RubberDub # 7minutes , ' Merson put on Twitter . Merson initially angered Townsend for writing in his Sky Sports column that 'if Andros Townsend can get in ( the England team ) then it opens it up to anybody . ' Paul Merson had another dig at Andros Townsend after his appearance for Tottenham against Burnley Townsend was brought on in the 83rd minute for Tottenham as they drew 0-0 against Burnley Andros Townsend scores England 's equaliser in their 1-1 friendly draw with Italy in Turin on Tuesday night The former Arsenal man was proven wrong when Townsend hit a stunning equaliser for England against Italy and he duly admitted his mistake . 'It 's not as though I was watching hoping he would n't score for England , I 'm genuinely pleased for him and fair play to him \u00e2\u20ac\u201c it was a great goal , ' Merson said . 'It 's just a matter of opinion , and my opinion was that he got pulled off after half an hour at Manchester United in front of Roy Hodgson , so he should n't have been in the squad . 'When I 'm wrong , I hold my hands up . I do n't have a problem with doing that - I 'll always be the first to admit when I 'm wrong . ' Townsend hit back at Merson on Twitter after scoring for England against Italy Sky Sports pundit Merson ( centre ) criticised Townsend 's call-up to the England squad last week Townsend hit back at Merson after netting for England in Turin on Wednesday , saying 'Not bad for a player that should be 'nowhere near the squad ' ay @ PaulMerse ? ' Any bad feeling between the pair seemed to have passed but Merson was unable to resist having another dig at Townsend after Tottenham drew at Turf Moor .",
        summary="Paul merson was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley . Andros townsend scored the tottenham midfielder in the 89th minute . Paul merson had another dig at andros townsend after his appearance . The midfielder had been brought on to the england squad last week . Click here for all the latest arsenal news news .",
    )

    g_evaluator = GEvalJudgeEvaluator(llm_evaluator=g_eval)

    response = g_evaluator(input_strs=[input_str])
    print(f"response: {response}")


# from deepeval import evaluate
# from deepeval.metrics import ContextualRecallMetric
# from deepeval.test_case import LLMTestCase
# from adalflow.utils import setup_env

# setup_env()

# # Replace this with the actual output from your LLM application
# actual_output = "We offer a 30-day full refund at no extra cost."

# # Replace this with the expected output from your RAG generator
# expected_output = "You are eligible for a 30 day full refund at no extra cost."

# # Replace this with the actual retrieved context from your RAG pipeline
# retrieval_context = [
#     "All customers are eligible for a 30 day full refund at no extra cost."
# ]

# metric = ContextualRecallMetric(threshold=0.7, model="gpt-4", include_reason=True)
# test_case = LLMTestCase(
#     input="What if these shoes don't fit?",
#     actual_output=actual_output,
#     expected_output=expected_output,
#     retrieval_context=retrieval_context,
# )

# print(f"metric: {metric}")

# metric.measure(test_case)
# print(metric.score)
# print(metric.reason)

# # or evaluate test cases in bulk
# evaluate([test_case], [metric])
