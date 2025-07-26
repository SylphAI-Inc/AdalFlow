# """
# AdalFlow RAG Agent Runner using AdalFlow's Runner and Agent components

# This implementation demonstrates how to create a RAG (Retrieval-Augmented Generation)
# system using AdalFlow's core Agent and Runner components, with a DspyRetriever
# for document retrieval functionality.

# Includes comprehensive evaluation using:
# - Answer accuracy evaluation (exact match, F1 score)
# - Retrieval quality evaluation (Recall@k, Precision@k)
# - End-to-end RAG system performance metrics
# """

# from typing import Optional, List, Dict, Any
# import dspy
# from dataclasses import dataclass

# print(dspy.__version__)

# from adalflow.components.agent import Agent, Runner
# from adalflow.core.func_tool import FunctionTool
# from adalflow.components.model_client.openai_client import OpenAIClient
# from adalflow.utils import setup_env
# from adalflow.eval.answer_match_acc import AnswerMatchAcc
# from adalflow.eval.retriever_recall import RetrieverEvaluator

# # Importing DspyRetriever directly to avoid config dependencies
# from adalflow.core.retriever import Retriever
# from adalflow.core.types import RetrieverOutput, ToolOutput
# from adalflow.datasets import HotPotQA




# @dataclass
# class RAGEvaluationResult:
#     """Comprehensive evaluation results for RAG system."""
#     # Answer evaluation metrics
#     exact_match_accuracy: float
#     f1_score: float
    
#     # Retrieval evaluation metrics
#     retrieval_recall: float
#     retrieval_precision: float
    
#     # Per-item detailed results
#     per_item_answer_scores: List[float]
#     per_item_retrieval_metrics: List[Dict[str, float]]
    
#     # System performance
#     avg_steps_per_query: float
#     total_queries: int


# # DspyRetriever class definition (copied to avoid config dependencies)
# class DspyRetriever(Retriever):
#     def __init__(self, top_k: int = 3):
#         super().__init__()
#         self.top_k = top_k
#         self.dspy_retriever = dspy.Retrieve(k=top_k)

#     def call(
#         self, input: str, top_k: Optional[int] = None, id: str = None
#     ) -> RetrieverOutput:
#         """Retrieves the top k passages using input as the query.
#         Ensure you get all the context to answer the original question.
        
#         Args:
#             input: The search query string
#             top_k: Number of documents to retrieve (overrides self.top_k)
#             id: Optional identifier for tracking

#         Returns:
#             RetrieverOutput with retrieved document passages
#         """

#         k = top_k or self.top_k

#         if not input:
#             raise ValueError(f"Input cannot be empty, top_k: {k}")

#         output = self.dspy_retriever(query=input, k=k)
#         documents = output.passages

#         return RetrieverOutput(
#             query=input,
#             documents=documents,
#             doc_indices=[],
#         )
    
#     def get_retrieved_contexts(self, input: str, top_k: Optional[int] = None) -> List[str]:
#         """Get retrieved contexts as a list of strings for evaluation."""
#         result = self.call(input, top_k)
#         return result.documents


# # Configure DSPy retriever
# colbertv2_wiki17_abstracts = dspy.ColBERTv2(
#     url="http://20.102.90.50:2017/wiki17_abstracts"
# )
# dspy.settings.configure(rm=colbertv2_wiki17_abstracts)


# def create_rag_system():
#     """Create and configure the RAG system components."""
#     # Initialize the retriever
#     dspy_retriever = DspyRetriever(top_k=3)

#     # Create tools for the agent
#     tools = [
#         FunctionTool(dspy_retriever.call),
#     ]

#     # Create the agent
#     agent = Agent(
#         name="RAGAgent",
#         tools=tools,
#         model_client=OpenAIClient(),
#         model_kwargs={
#             "model": "gpt-4o",
#             "temperature": 0.3,
#         },
#         answer_data_type=str,
#         max_steps=5,
#     )

#     # Create the runner
#     runner = Runner(agent=agent, max_steps=5)
    
#     return agent, runner, dspy_retriever

# def evaluate_retrieval_only(num_samples: int = 10) -> Dict:
#     """Evaluate retrieval performance without agent reasoning.
    
#     This directly compares what the dataset says should be retrieved
#     vs what the retriever actually returns - no agent involved.
#     """
#     print(f"\n=== Pure Retrieval Evaluation (n={num_samples}) ===")
#     print("Note: This evaluates retrieval quality WITHOUT agent reasoning")
    
#     # Load dataset
#     dataset = HotPotQA(split="train", size=num_samples)
    
#     # Create retriever
#     retriever = DspyRetriever(top_k=3)
    
#     # Get ground truth from dataset (what should be retrieved)
#     gt_contexts = []
#     retrieved_contexts = []
    
#     print("Processing retrieval-only evaluation...")
#     for i, sample in enumerate(dataset):
#         print(f"\rSample {i+1}/{num_samples}", end="")
        
#         # Ground truth: gold titles from dataset
#         gt_titles = list(sample.gold_titles) if hasattr(sample, 'gold_titles') else []
#         gt_contexts.append(gt_titles)
        
#         # Retrieved: what our retriever actually gets
#         retrieved = retriever.get_retrieved_contexts(sample.question)
#         retrieved_contexts.append(retrieved)
    
#     print("\nEvaluating retrieval quality...")
    
#     # Evaluate retrieval quality
#     retrieval_evaluator = RetrieverEvaluator()
#     result = retrieval_evaluator.compute(retrieved_contexts, gt_contexts)
    
#     return result


# def evaluate_rag_system(num_samples: int = 10) -> RAGEvaluationResult:
#     """Comprehensive evaluation of the RAG system using multiple metrics.
    
#     Args:
#         num_samples: Number of samples to evaluate on
        
#     Returns:
#         RAGEvaluationResult with comprehensive metrics
#     """
#     print(f"\n=== RAG System Evaluation (n={num_samples}) ===")
    
#     # Load evaluation dataset
#     dataset = HotPotQA(split="train", size=num_samples)
    
#     # Create RAG system
#     agent, runner, retriever = create_rag_system()
#     agent.eval()  # Set to evaluation mode
    
#     # Initialize evaluators
#     exact_match_evaluator = AnswerMatchAcc(type="exact_match")
#     f1_evaluator = AnswerMatchAcc(type="f1_score")
#     retrieval_evaluator = RetrieverEvaluator()
    
#     # Collect predictions and ground truth
#     predictions = []
#     ground_truth_answers = []
#     retrieved_contexts_list = []
#     ground_truth_contexts_list = []
#     steps_per_query = []
    
#     print("Processing queries...")
#     for i, sample in enumerate(dataset):
#         print(f"\rQuery {i+1}/{num_samples}", end="")
        
#         # Get agent's answer
#         result = runner.call(
#             prompt_kwargs={"input_str": sample.question}
#         )
        
#         predictions.append(result.answer)
#         ground_truth_answers.append(sample.answer)
#         steps_per_query.append(len(result.step_history))
        
#         # Get retrieved contexts for evaluation
#         retrieved_contexts = retriever.get_retrieved_contexts(sample.question)
#         retrieved_contexts_list.append(retrieved_contexts)
        
#         # Ground truth contexts (from HotPotQA gold titles)
#         gt_contexts = list(sample.gold_titles) if hasattr(sample, 'gold_titles') else []
#         ground_truth_contexts_list.append(gt_contexts)
    
#     print("\nEvaluating results...")
    
#     # Answer evaluation
#     exact_match_result = exact_match_evaluator.compute(predictions, ground_truth_answers)
#     f1_result = f1_evaluator.compute(predictions, ground_truth_answers)
    
#     # Retrieval evaluation (only if we have ground truth contexts)
#     retrieval_result = None
#     if any(gt_contexts for gt_contexts in ground_truth_contexts_list):
#         retrieval_result = retrieval_evaluator.compute(
#             retrieved_contexts_list, ground_truth_contexts_list
#         )
    
#     # Create comprehensive result
#     evaluation_result = RAGEvaluationResult(
#         exact_match_accuracy=exact_match_result.avg_score,
#         f1_score=f1_result.avg_score,
#         retrieval_recall=retrieval_result["avg_recall"] if retrieval_result else 0.0,
#         retrieval_precision=retrieval_result["avg_precision"] if retrieval_result else 0.0,
#         per_item_answer_scores=exact_match_result.per_item_scores,
#         per_item_retrieval_metrics=retrieval_result["recall_list"] if retrieval_result else [],
#         avg_steps_per_query=sum(steps_per_query) / len(steps_per_query),
#         total_queries=num_samples,
#     )
    
#     return evaluation_result


# def print_evaluation_results(result: RAGEvaluationResult):
#     """Print formatted evaluation results."""
#     print("\n" + "="*60)
#     print("RAG SYSTEM EVALUATION RESULTS")
#     print("="*60)
    
#     print(f"\n📊 Answer Quality Metrics:")
#     print(f"  • Exact Match Accuracy: {result.exact_match_accuracy:.1%}")
#     print(f"  • F1 Score:             {result.f1_score:.1%}")
    
#     if result.retrieval_recall > 0:
#         print(f"\n🔍 Retrieval Quality Metrics:")
#         print(f"  • Recall@k:    {result.retrieval_recall:.1%}")
#         print(f"  • Precision@k: {result.retrieval_precision:.1%}")
    
#     print(f"\n⚡ System Performance:")
#     print(f"  • Average steps per query: {result.avg_steps_per_query:.1f}")
#     print(f"  • Total queries evaluated: {result.total_queries}")
    
#     # Distribution analysis
#     if result.per_item_answer_scores:
#         perfect_answers = sum(1 for score in result.per_item_answer_scores if score == 1.0)
#         print(f"  • Perfect answers:         {perfect_answers}/{result.total_queries} ({perfect_answers/result.total_queries:.1%})")
    
#     print("\n" + "="*60)


# def print_retrieval_results(result: Dict):
#     """Print retrieval-only evaluation results."""
#     print("\n" + "="*50)
#     print("PURE RETRIEVAL EVALUATION RESULTS")
#     print("(No Agent Reasoning - Direct Retrieval Only)")
#     print("="*50)
    
#     print(f"\n🔍 Retrieval Quality Metrics:")
#     print(f"  • Recall@k:    {result['avg_recall']:.1%}")
#     print(f"  • Precision@k: {result['avg_precision']:.1%}")
#     print(f"  • Top-k:       {result['top_k']}")
    
#     print(f"\n📈 Per-Query Distribution:")
#     recall_list = result['recall_list']
#     perfect_recalls = sum(1 for r in recall_list if r == 1.0)
#     zero_recalls = sum(1 for r in recall_list if r == 0.0)
    
#     print(f"  • Perfect recall (1.0):  {perfect_recalls}/{len(recall_list)}")
#     print(f"  • Zero recall (0.0):     {zero_recalls}/{len(recall_list)}")
#     print(f"  • Partial recall:        {len(recall_list) - perfect_recalls - zero_recalls}/{len(recall_list)}")
    
#     print("\n" + "="*50)


# def main():
#     """Main execution function with comprehensive evaluation."""
#     # Setup environment variables
#     setup_env()
    
#     print("AdalFlow RAG Agent Runner - Comprehensive Evaluation Demo")
#     print("=========================================================\n")
    
#     # 1. Pure retrieval evaluation (no agent)
#     retrieval_result = evaluate_retrieval_only(num_samples=15)
#     print_retrieval_results(retrieval_result)
    
#     # 2. Full RAG system evaluation (with agent)
#     rag_result = evaluate_rag_system(num_samples=15)
#     print_evaluation_results(rag_result)
    
#     # 3. Comparison summary
#     print("\n" + "="*60)
#     print("COMPARISON: RETRIEVAL vs FULL RAG SYSTEM")
#     print("="*60)
#     print(f"Pure Retrieval Recall:    {retrieval_result['avg_recall']:.1%}")
#     print(f"RAG System Answer Acc:    {rag_result.exact_match_accuracy:.1%}")
#     print(f"RAG System F1 Score:      {rag_result.f1_score:.1%}")
#     print("\nNote: Pure retrieval measures document relevance,")
#     print("      RAG system measures final answer quality.")
#     print("="*60)
    
#     return rag_result, retrieval_result


# if __name__ == "__main__":
#     rag_result, retrieval_result = main()
    
#     # Optional: Save results for further analysis
#     # import json
#     # results = {
#     #     'rag_system': rag_result.__dict__,
#     #     'pure_retrieval': retrieval_result
#     # }
#     # with open('rag_evaluation_results.json', 'w') as f:
#     #     json.dump(results, f, indent=2)
#     #     print("\nResults saved to rag_evaluation_results.json")
