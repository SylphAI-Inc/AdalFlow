"""This contains different evaluators and their associated metrics."""

from typing import List
from lightrag.core.generator import Generator
from lightrag.components.model_client import OpenAIClient


DEFAULT_LLM_EVALUATOR_PROMPT = r"""
<<SYS>>{# task desc #}
{% if task_desc_str %}
{{task_desc_str}}
{% endif %}<</SYS>>
---------------------
{# question #}
Question: {{question_str}}
{# ground truth answer #}
Ground truth answer: {{gt_answer_str}}
{# predicted answer #}
Predicted answer: {{pred_answer_str}}
{# judgement question #}
Judgement question: {{judgement_str}}
{# assistant response #}
You:
"""

DEFAULT_LLM_EVALUATOR = Generator(
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-3.5-turbo", "temperature": 0.3, "stream": False},
    template=DEFAULT_LLM_EVALUATOR_PROMPT,
    preset_prompt_kwargs={
        "task_desc_str": r"""
            You are a helpful assistant.
            Given the question, ground truth answer, and predicted answer, you need to answer the judgement query.
            Output True or False according to the judgement query.
            """
    },
)


class LLMasJudge:
    r"""
    LLM as judge for evaluating the performance of a LLM.
    """

    def __init__(self, llm_evaluator: Generator = DEFAULT_LLM_EVALUATOR):
        r"""
        Initialize a new instance of LLMasJudge.

        Args:
            llm_evaluator (Generator): LLM model to be used as judge.
        """
        self.llm_evaluator = llm_evaluator

    def compute_judgement_single_question(
        self, question: str, pred_answer: str, gt_answer: str, judgement_query: str
    ) -> bool:
        r"""
        Get the judgement of the predicted answer for a single question.

        Args:
            question (str): Question string.
            pred_answer (str): Predicted answer string.
            gt_answer (str): Ground truth answer string.
            judgement_query (str): Judgement query string.

        Returns:
            bool: Judgement result.
        """
        output = self.llm_evaluator(
            prompt_kwargs={
                "judgement_str": judgement_query,
                "question_str": question,
                "gt_answer_str": gt_answer,
                "pred_answer_str": pred_answer,
            },
        )
        judgement = output.raw_response
        if judgement == "True":
            return True
        elif judgement == "False":
            return False
        else:
            raise ValueError(f"Invalid judgement: {judgement}")

    def compute_judgement(
        self,
        all_questions: List[str],
        all_pred_answer: List[str],
        all_gt_answer: List[str],
        judgement_query: str,
    ) -> List[bool]:
        r"""
        Get the judgement of the predicted answer for a list of questions.

        Args:
            all_questions (List[str]): List of question strings.
            all_pred_answer (List[str]): List of predicted answer strings.
            all_gt_answer (List[str]): List of ground truth answer strings.
            judgement_query (str): Judgement query string.

        Returns:
            List[bool]: Judgement results.
        """
        judgement_list = []
        for question, pred_answer, gt_answer in zip(
            all_questions, all_pred_answer, all_gt_answer
        ):
            judgement = self.compute_judgement_single_question(
                question, pred_answer, gt_answer, judgement_query
            )
            judgement_list.append(judgement)

        return judgement_list.count(True) / len(judgement_list), judgement_list
