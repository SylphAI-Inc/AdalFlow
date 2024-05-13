from typing import List, Union, Tuple
from core.tokenizer import Tokenizer
from core.generator import Generator


DEFAULT_LLM_EVALUATOR_PROMPT = r"""
<<SYS>>{# task desc #}
{% if task_desc_str %}
{{task_desc_str}}
{% endif %}<</SYS>>
---------------------
{# question #}
Question: {{question_str}}
{# ground truth answer #}
Ground truth answer: {{gt_answer}}
{# predicted answer #}
Predicted answer: {{pred_answer}}
{# judgement question #}
Judgement question: {{query_str}}
{# assistant response #}
You:
"""


class AnswerMacthEvaluator:
    """
    Evaluator for evaluating the performance of a matching model.
    Args:
        type (str): Type of matching evaluation. Can be "exact_match" or "fuzzy_match". "exact_match" requires the predicted answer to be exactly the same as the ground truth answer. "fuzzy_match" requires the predicted answer to contain the ground truth answer.
    """

    def __init__(self, type: str = "exact_match"):
        self.type = type

    def compute_match_acc_single_query(self, pred_answer: str, gt_answer: str) -> float:
        """
        Compute the match accuracy of the predicted answer for a single query.
        Args:
            pred_answer (str): Predicted answer string
            gt_answer (str): Ground truth answer string
        Returns:
            float: Match accuracy
        """
        if self.type == "exact_match":
            return 1.0 if pred_answer == gt_answer else 0.0
        elif self.type == "fuzzy_match":
            return 1.0 if gt_answer in pred_answer else 0.0
        else:
            raise NotImplementedError

    def compute_match_acc(
        self, all_pred_answer: List[str], all_gt_answer: List[str]
    ) -> Tuple[float, List[float]]:
        """
        Compute the match accuracy of the predicted answer for a list of queries.
        Args:
            all_pred_answer: List of predicted answer strings
            all_gt_answer: List of ground truth answer strings
        Returns:
            float: Average match accuracy
            List[float]: Match accuracy values for each query
        """
        match_acc_list = []
        for pred_answer, gt_answer in zip(all_pred_answer, all_gt_answer):
            match = self.compute_match_acc_single_query(pred_answer, gt_answer)
            match_acc_list.append(match)

        return sum(match_acc_list) / len(match_acc_list), match_acc_list


class RetrieverEvaluator:
    """
    Evaluator for evaluating the performance of a retriever.
    """

    def __init__(self):
        pass

    def compute_recall_single_query(
        self, retrieved_context: str, gt_context: Union[str, List[str]]
    ) -> float:
        """
        Compute the recall of the retrieved context for a single query.
        Args:
            retrieved_context (str): Retrieved context string
            gt_context (Union[str, List[str]]): Context string or list of context strings to compare against
        Returns:
            float: Recall value
        """
        if isinstance(gt_context, str):
            gt_context = [gt_context]
        recalled = 0
        for gt_context_sentence in gt_context:
            if gt_context_sentence in retrieved_context:
                recalled += 1
        return recalled / len(gt_context)

    def compute_recall(
        self,
        all_retrieved_context: List[str],
        all_gt_context: Union[List[str], List[List[str]]],
    ) -> Tuple[float, List[float]]:
        """
        Compute the recall of the retrieved context for a list of queries.
        Args:
            all_retrieved_context: List of retrieved context strings
            all_gt_context: List of ground truth context strings and each of them can be a string or a list of strings
        Returns:
            float: Average recall value
            List[float]: Recall values for each query
        """
        recall_list = []
        for retrieved_context, gt_context in zip(all_retrieved_context, all_gt_context):
            recall = self.compute_recall_single_query(retrieved_context, gt_context)
            recall_list.append(recall)

        return sum(recall_list) / len(recall_list), recall_list

    def compute_context_relevance_single_query(
        self, retrieved_context: str, gt_context: Union[str, List[str]]
    ) -> float:
        """
        Compute the context relevance of the retrieved context for a single query. The context relevance is the ratio of the number of relevant context tokens in the retrieved context to the total number of tokens in the retrieved context.
        Args:
            retrieved_context (str): Retrieved context string
            gt_context (Union[str, List[str]]): Context string or list of context strings to compare against
        Returns:
            float: Context relevance value
        """
        if isinstance(gt_context, str):
            gt_context = [gt_context]
        relevant_tokens = 0
        tokenizer = Tokenizer()
        for gt_context_sentence in gt_context:
            if gt_context_sentence in retrieved_context:
                relevant_tokens += tokenizer.count_tokens(gt_context_sentence)
        return relevant_tokens / tokenizer.count_tokens(retrieved_context)

    def compute_context_relevance(
        self,
        all_retrieved_context: List[str],
        all_gt_context: Union[List[str], List[List[str]]],
    ) -> Tuple[float, List[float]]:
        """
        Compute the context relevance of the retrieved context for a list of queries. The context relevance is the ratio of the number of relevant context tokens in the retrieved context to the total number of tokens in the retrieved context.
        Args:
            all_retrieved_context: List of retrieved context strings
            all_gt_context: List of ground truth context strings and each of them can be a string or a list of strings
        Returns:
            float: Average context relevance value
            List[float]: Context relevance values for each query
        """
        context_relevance_list = []
        for retrieved_context, gt_context in zip(all_retrieved_context, all_gt_context):
            context_relevance = self.compute_context_relevance_single_query(
                retrieved_context, gt_context
            )
            context_relevance_list.append(context_relevance)

        return (
            sum(context_relevance_list) / len(context_relevance_list),
            context_relevance_list,
        )


class LLMasJudge:
    """
    LLM as judge for evaluating the performance of a LLM.

    Args:
        llm_evaluator (Generator): LLM model to be used as judge
    """

    def __init__(self, llm_evaluator: Generator):
        self.llm_evaluator = llm_evaluator

    def compute_judgement_single_question(
        self, question: str, pred_answer: str, gt_answer: str, judgement_query: str
    ) -> bool:
        """
        Get the judgement of the predicted answer for a single question.
        Args:
            question (str): Question string
            pred_answer (str): Predicted answer string
            gt_answer (str): Ground truth answer string
            judgement_query (str): judgement query string
        Returns:
            bool: Judgement result
        """
        judgement = self.llm_evaluator(
            input=judgement_query,
            prompt_kwargs={
                "question_str": question,
                "gt_answer": gt_answer,
                "pred_answer": pred_answer,
            },
        )
        return judgement["judgement"]

    def compute_judgement(
        self,
        all_questions: List[str],
        all_pred_answer: List[str],
        all_gt_answer: List[str],
        judgement_query: str,
    ) -> List[bool]:
        """
        Get the judgement of the predicted answer for a list of questions.
        Args:
            all_questions: List of question strings
            all_pred_answer: List of predicted answer strings
            all_gt_answer: List of ground truth answer strings
            judgement_query (str): judgement query string
        Returns:
            List[bool]: Judgement results
        """
        judgement_list = []
        for question, pred_answer, gt_answer in zip(
            all_questions, all_pred_answer, all_gt_answer
        ):
            judgement = self.compute_judgement_single_question(
                question, pred_answer, gt_answer, judgement_query
            )
            judgement_list.append(judgement)

        return judgement_list
