"""This is the metric to use an LLM as a judge for evaluating the performance of predicted answers."""

from typing import List, Dict, Any, Optional
import logging


from lightrag.core.generator import Generator
from lightrag.core.component import Component
from lightrag.core.model_client import ModelClient


log = logging.getLogger(__name__)

DEFAULT_LLM_EVALUATOR_PROMPT = r"""
<<SYS>>{# task desc #}
You are a helpful assistant.
Given the question, ground truth answer, and predicted answer, you need to answer the judgement query.
Output True or False according to the judgement query.<</SYS>>
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


# print(f"globals: {globals()}")

DEFAULT_LLM_EVALUATOR_MODEL_KWARGS = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.3,
    "stream": False,
}


class DefaultLLMJudge(Component):
    __doc__ = r"""Demonstrate how to use an LLM/Generator to output True or False for a judgement query.

    You can use any of your template to adapt to more tasks and sometimes you can directly ask LLM to output a score in range [0, 1] instead of only True or False.

    A call on the LLM judge equalize to _compute_single_item method.

    Args:
        model_client (ModelClient): The model client to use for the generator.
        model_kwargs (Dict[str, Any], optional): The model kwargs to pass to the model client. Defaults to {}. Please refer to :ref:`ModelClient<components-model_client>` for the details on how to set the model_kwargs for your specific model if it is from our library.
    """

    def __init__(
        self,
        model_client: Optional[ModelClient] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model_client = model_client
        if model_client is None:
            log.info("model_client is None, default to OpenAIClient.")
            try:
                from lightrag.components.model_client import OpenAIClient
            except ImportError:
                raise ImportError(
                    "OpenAIClient is not available. Please fix the import error or set your own choice of model_client and model_kwargs."
                )
            self.model_client = OpenAIClient()
        self.model_kwargs = model_kwargs or DEFAULT_LLM_EVALUATOR_MODEL_KWARGS
        self.llm_evaluator = Generator(
            model_client=self.model_client,
            model_kwargs=self.model_kwargs,
            template=DEFAULT_LLM_EVALUATOR_PROMPT,
        )

    def call(
        self, question: str, gt_answer: str, pred_answer: str, judgement_query: str
    ) -> bool:
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
                "judgement_str": judgement_query,
            }
        )

        judgement = output.raw_response
        judgement = judgement.strip().lower()
        if "true" in judgement:
            return True
        elif "false" in judgement:
            return False
        else:
            raise ValueError(f"Invalid judgement: {judgement}")


class LLMasJudge:
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
    """

    def __init__(
        self,
        llm_evaluator: Optional[Component] = None,
    ):
        self.llm_evaluator = llm_evaluator or DefaultLLMJudge()

    def compute(
        self,
        questions: List[str],
        gt_answers: List[str],
        pred_answers: List[str],
        judgement_query: str,
    ) -> List[bool]:
        r"""
        Get the judgement of the predicted answer for a list of questions.

        Args:
            questions (List[str]): List of question strings.
            gt_answers (List[str]): List of ground truth answer strings.
            pred_answers (List[str]): List of predicted answer strings.
            judgement_query (str): Judgement query string.

        Returns:
            tuple:
                - float: Average judgement score.
                - List[bool]: Judgement results for each query.
        """
        judgement_list = []
        for question, gt_answer, pred_answer in zip(
            questions, gt_answers, pred_answers
        ):
            judgement = self.llm_evaluator(
                question, gt_answer, pred_answer, judgement_query
            )
            judgement_list.append(judgement)

        return judgement_list.count(True) / len(judgement_list), judgement_list


if __name__ == "__main__":

    questions = [
        "Is Beijing in China?",
        "Is Apple founded before Google?",
        "Is earth flat?",
    ]
    pred_answers = ["Yes", "Yes, Appled is founded before Google", "Yes"]
    gt_answers = ["Yes", "Yes", "No"]
    judgement_query = (
        "For the question, does the predicted answer contain the ground truth answer?"
    )
    llm_judge = LLMasJudge()
    avg_judgement, judgement_list = llm_judge.compute(
        questions, gt_answers, pred_answers, judgement_query
    )
    print(avg_judgement)
    print(judgement_list)
