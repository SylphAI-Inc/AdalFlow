"""This is the metric to use an LLM as a judge for evaluating the performance of predicted answers."""

from typing import List, Dict, Any
from lightrag.core.generator import Generator
from lightrag.core.model_client import ModelClient

try:
    from lightrag.components.model_client import OpenAIClient
except ImportError:
    pass

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

# We use OpenAIClient as the default LLM evaluator if it is available. Otherwise, we set it to None.
if "OpenAIClient" in globals():
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
else:
    DEFAULT_LLM_EVALUATOR = None


class LLMasJudge:
    r"""
    LLM as judge for evaluating the performance of a LLM.

    Args:
        llm_evaluator (Generator): The LLM evaluator to use as a judge. You can use the default LLM evaluator (`DEFAULT_LLM_EVALUATOR`) or set your own LLM evaluator. The task description of the DEFAULT_LLM_EVALUATOR is as follows: "You are a helpful assistant. Given the question, ground truth answer, and predicted answer, you need to answer the judgement query. Output True or False according to the judgement query." The model client and model kwargs of the DEFAULT_LLM_EVALUATOR are set to OpenAIClient and {"model": "gpt-3.5-turbo", "temperature": 0.3, "stream": False}, respectively, but you can set your own model client and model kwargs if you want.
        model_client (ModelClient): If you want to use a different ModelClient instead of the default OpenAIClient for the default LLM evaluator, you can set it here.
        model_kwargs (Dict[str, Any]): If you want to use different model kwargs instead of the default model kwargs for the default LLM evaluator, you can set them here.

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
        questions, pred_answers, gt_answers, judgement_query
        )
        >>> avg_judgement
        2 / 3
        >>> judgement_list
        [True, True, False]
    """

    def __init__(
        self,
        llm_evaluator: Generator = DEFAULT_LLM_EVALUATOR,
        model_client: ModelClient = None,
        model_kwargs: Dict[str, Any] = None,
    ):
        if llm_evaluator is not DEFAULT_LLM_EVALUATOR:
            self.llm_evaluator = llm_evaluator
        else:
            if model_client is None and model_kwargs is None:
                self.llm_evaluator = llm_evaluator
            elif model_client is not None and model_kwargs is not None:
                self.llm_evaluator = Generator(
                    model_client=model_client,
                    model_kwargs=model_kwargs,
                    template=DEFAULT_LLM_EVALUATOR_PROMPT,
                    preset_prompt_kwargs={
                        "task_desc_str": r"""
                            You are a helpful assistant.
                            Given the question, ground truth answer, and predicted answer, you need to answer the judgement query.
                            Output True or False according to the judgement query.
                            """
                    },
                )
            else:
                raise ValueError(
                    "model_client and model_kwargs should be both None or both not None."
                )
            assert self.llm_evaluator is not None, "llm_evaluator should not be None."

    def _compute_single_item(
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

    def compute(
        self,
        questions: List[str],
        pred_answers: List[str],
        gt_answers: List[str],
        judgement_query: str,
    ) -> List[bool]:
        r"""
        Get the judgement of the predicted answer for a list of questions.

        Args:
            questions (List[str]): List of question strings.
            pred_answers (List[str]): List of predicted answer strings.
            gt_answers (List[str]): List of ground truth answer strings.
            judgement_query (str): Judgement query string.

        Returns:
            List[bool]: Judgement results.
        """
        judgement_list = []
        for question, pred_answer, gt_answer in zip(
            questions, pred_answers, gt_answers
        ):
            judgement = self._compute_single_item(
                question, pred_answer, gt_answer, judgement_query
            )
            judgement_list.append(judgement)

        return judgement_list.count(True) / len(judgement_list), judgement_list
