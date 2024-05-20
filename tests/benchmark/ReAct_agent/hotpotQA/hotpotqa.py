"""
https://arxiv.org/abs/2210.03629, published in Mar, 2023

Use LightRAG agent component to build the same version of ReAct agent as in the paper.
Apply the similar code for wikipedia search from the Paper (open-source).
Use the default prompt without any example.
"""


import dotenv
from components.api_client.openai_client import OpenAIClient
from components.agent.react_agent import ReActAgent
from core.tool_helper import FunctionTool
from core.api_client import APIClient
import time
from tests.benchmark.ReAct_agent.hotpotQA.tools import search, lookup
from eval.evaluator import AnswerMacthEvaluator


# load evironment
dotenv.load_dotenv(dotenv_path=".env", override=True)

# # paper example
# prompt_file = './tests/benchmark/ReAct_agent/paper_data/prompts_naive.json'
# with open(prompt_file, 'r') as f:
#     prompt_dict = json.load(f)
# webthink_examples = prompt_dict['webthink_simple6']

# # paper instruction prompt -> should be sys prompt
# instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
# (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
# (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
# (3) Finish[answer], which returns the answer and finishes the task.
# Here are some examples.
# """

from components.api_client import GroqAPIClient
tools = [
        FunctionTool.from_defaults(fn=search),
        FunctionTool.from_defaults(fn=lookup),
    ]

llm_model_kwargs = {
        "model": "llama3-70b-8192",  # llama3 is not good with string formatting, llama3 8b is also bad at following instruction, 70b is better but still not as good as gpt-3.5-turbo
        # mistral also not good: mixtral-8x7b-32768, but with better prompt, it can still work
        "temperature": 0.0,
    }

# TODO: Add examples to improve the performance
# examples = [] 

react_agent = ReActAgent(
    # examples=examples,
    tools=tools,
    # max_steps=5,
    model_client=GroqAPIClient,
    model_kwargs=llm_model_kwargs,
)

import json
evaluator = AnswerMacthEvaluator(type="exact_match")
file = open('./tests/benchmark/ReAct_agent/paper_data/hotpot_dev_v1_simplified.json')
dataset = json.load(file)

average_time = 0
num_questions = 10
em = 0
for i in range(num_questions):
    question = dataset[i].get("question")
    gt_answer = dataset[i].get("answer")
    # print(question)
    # print(gt_answer)
    
    t0 = time.time()
    pred_answer = react_agent(question)
    
    res = evaluator.compute_match_acc_single_query(pred_answer=pred_answer, gt_answer=gt_answer)
    em += res
    
    average_time += time.time() - t0
    # print(f"Answer: {pred_answer}")
    # print(f'Result: {res}')
print(f"Average time: {average_time / num_questions}")
print(f"EM: {em / num_questions}")

"""
Average time: 150.9188715696335
EM: 0.1
"""