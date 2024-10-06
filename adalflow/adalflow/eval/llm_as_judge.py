"""This is the metric to use an LLM as a judge for evaluating the performance of predicted answers."""

from typing import List, Dict, Any, Optional, TYPE_CHECKING, Union, Literal, Tuple
from dataclasses import dataclass
import logging
from itertools import zip_longest


if TYPE_CHECKING:
    pass
from adalflow.core.component import Component
from adalflow.optim.parameter import Parameter, ParameterType
from adalflow.core.model_client import ModelClient
from adalflow.eval.base import BaseEvaluator
from adalflow.eval.functional import confidence_interval

__all__ = ["DefaultLLMJudge", "LLMasJudge", "LLMJudgeEvalResult"]

log = logging.getLogger(__name__)

DEFAULT_LLM_EVALUATOR_PROMPT = r"""<START_OF_SYSTEM_PROMPT>
{# task desc #}
{{task_desc_str}}
{# examples #}
{% if examples_str %}
{{examples_str}}
{% endif %}
<END_OF_SYSTEM_PROMPT>
---------------------
<START_OF_USER>
{# question #}
{% if question_str is defined %}
Question: {{question_str}}
{% endif %}
{# ground truth answer #}
{% if gt_answer_str is defined %}
Ground truth answer: {{gt_answer_str}}
{% endif %}
{# predicted answer #}
Predicted answer: {{pred_answer_str}}
<END_OF_USER>
"""

DEFAULT_JUDGEMENT_QUERY = "Does the predicted answer contain the ground truth answer? Say True if yes, False if no."


DEFAULT_LLM_EVALUATOR_MODEL_KWARGS = {
    "model": "gpt-3.5-turbo",
    "temperature": 1,
    "stream": False,
}


@dataclass
class LLMJudgeEvalResult:
    avg_score: float
    judgement_score_list: List[bool]
    confidence_interval: Tuple[float, float]


class DefaultLLMJudge(Component):
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
        jugement_query: Optional[str] = None,
        example_str: Optional[str] = None,
        output_type: Literal["bool", "float"] = "bool",
        use_cache: bool = True,
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
        self.template = template or DEFAULT_LLM_EVALUATOR_PROMPT
        self._jugement_query = jugement_query or DEFAULT_JUDGEMENT_QUERY
        self.llm_evaluator = Generator(
            model_client=self.model_client,
            model_kwargs=self.model_kwargs,
            template=self.template,
            use_cache=use_cache,
            prompt_kwargs={
                "task_desc_str": Parameter(
                    data=f"""You are an evaluator. Given the question(optional), ground truth answer(optional), and predicted answer, {self._jugement_query}""",
                    param_type=ParameterType.PROMPT,
                ),
                "examples_str": Parameter(
                    data=example_str, param_type=ParameterType.DEMOS
                ),
            },
        )

        self.output_type = output_type

    def call(
        self,
        question: str,
        gt_answer: str,
        pred_answer: str,
    ) -> Union[bool, float]:
        r"""
        Get the judgement of the predicted answer for a single question.

        Args:
            question (str): Question string.
            gt_answer (str): Ground truth answer string.
            pred_answer (str): Predicted answer string.
            judgement_query (str): Judgement query string.

        Returns:
            bool: Judgement result.
        """
        output = self.llm_evaluator(
            prompt_kwargs={
                "question_str": question,
                "gt_answer_str": gt_answer,
                "pred_answer_str": pred_answer,
            }
        )

        judgement = output.raw_response
        judgement = judgement.strip().lower()
        output = False if self.output_type == "bool" else 0.0
        if "true" in judgement:
            output = True if self.output_type == "bool" else 1.0
        elif "false" in judgement:
            output = False if self.output_type == "bool" else 0.0
        else:
            print(f"Invalid judgement: {judgement}, use False or 0.0 instead.")
            # raise ValueError(f"Invalid judgement: {judgement}")
        return output

    def _extra_repr(self) -> str:
        s = f"judgement_query= {self._jugement_query}, "
        return s


class LLMasJudge(BaseEvaluator):
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
        llm_judge: Optional[Component] = None,
    ):
        super().__init__()
        self.llm_judge = llm_judge or DefaultLLMJudge()

    def compute(
        self,
        *,
        pred_answers: List[str],
        questions: Optional[List[str]] = None,
        gt_answers: Optional[List[str]] = None,
        # judgement_query: Optional[str] = None,
    ) -> LLMJudgeEvalResult:
        r"""
        Get the judgement of the predicted answer for a list of questions.

        Args:
            questions (List[str]): List of question strings.
            gt_answers (List[str]): List of ground truth answer strings.
            pred_answers (List[str]): List of predicted answer strings.
            judgement_query (str): Judgement query string.

        Returns:
            LLMEvalResult: The evaluation result.

        """
        judgement_list = []
        questions = questions or [None] * len(pred_answers)
        gt_answers = gt_answers or [None] * len(pred_answers)

        for question, gt_answer, pred_answer in zip_longest(
            questions, gt_answers, pred_answers, fillvalue=None
        ):
            judgement = self.llm_judge(
                question,
                gt_answer,
                pred_answer,
            )
            judgement_list.append(judgement)

        avg_score = judgement_list.count(True) / len(judgement_list)

        judgement_score_list = [1 if judgement else 0 for judgement in judgement_list]
        confidence = confidence_interval(judgement_score_list)
        return LLMJudgeEvalResult(avg_score, judgement_score_list, confidence)

    def __str__(self) -> str:
        s = f"llm_judge={self.llm_judge}"
        return s
