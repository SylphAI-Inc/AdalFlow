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
from components.api_client import GroqAPIClient
import time
from tests.benchmark.ReAct_agent.hotpotQA.tools import search, lookup, normalize_answer
from eval.evaluator import AnswerMacthEvaluator
import logging
import json
from typing import List, Union, Callable, Optional, Any, Dict
from core.tool_helper import FunctionTool, AsyncCallable


logger = logging.getLogger(__name__)
logging.basicConfig(filename='./logs/fever.log', level=logging.INFO)

# load evironment
dotenv.load_dotenv(dotenv_path=".env", override=True)


# Reference: paper's instruction prompt. (we use our default DEFAULT_REACT_AGENT_SYSTEM_PROMPT)
# instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
# (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
# (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
# (3) Finish[answer], which returns the answer and finishes the task.
# Here are some examples.
# """


# setup examples for few-shot experiment
# 6 examples from the paper's source code(transformed the format to use in LightRAG)

FEVER_REACT_AGENT_SYSTEM_PROMPT = r"""
{# role/task description #}
You task is to determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. 
Your can only answer SUPPORTS, REFUTES or NOT ENOUGH INFORMATION, and nothing else.
{# REACT instructions #}
Each step you will read the previous Thought, Action, and Observation(execution result of the action)steps and then provide the next Thought and Action.

You only have access to the following tools:
{# tools #}
{% for tool in tools %}
{{ loop.index }}. ToolName: {{ tool.metadata.name }}
    Tool Description: {{ tool.metadata.description }}
    Tool Parameters: {{ tool.metadata.fn_schema_str }} {#tool args can be misleading, especially if we already have type hints and docstring in the function#}
{% endfor %}
{# output is always more robust to use json than string #}
---
Your output must be in valid JSON format(raw Python string format) with two keys:
{
    "thought": "<Why you are taking this action>",
    "action": "ToolName(<args>, <kwargs>)"
}
- Must double quote the JSON str.
- Inside of the JSON str, Must use escape double quote and escape backslash for string.
For example:
"action": "finish(\"John's.\")"
---
{# Specifications TODO: preference between the usage of llm tool vs the other tool #}
Process:
- Step 1: Read the user query and potentially divide it into subqueries. And get started with the first subquery.
- Call one available tool at a time to solve each subquery/subquestion. \
- At step 'finish', join all subqueries answers and finish the task.
Remember:
- Action must call one of the above tools with Took Name. It can not be empty.
- Read the Tool Description and ensure your args and kwarg follow what each tool expects in types. e.g. (a=1, b=2) if it is keyword argument or (1, 2) if it is positional.
- You will always end with 'finish' action to finish the task. The answer can be the final answer or failure message.
- When the initial query is simple, use minimum steps to answer the query.
{#Examples can be here#}
{# Check if there are any examples #}
{% if examples %}
<EXAMPLES>
{% for example in examples %}
{{ example }}
{% endfor %}
</EXAMPLES>
{% endif %}
<</SYS>>
-----------------
{# History #}
{% for history in step_history %}
Step {{history.step}}:
{
 "thought": "{{history.thought}}",
 "action": "{{history.action}}",
}
"observation": "{{history.observation}}"
{% endfor %}
"""

examples = [
    """
    Question: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
    Thought 1: I need to search Nikolaj Coster-Waldau and find if he has worked with the Fox Broadcasting Company.
    Action 1: search("Nikolaj Coster-Waldau")
    Observation 1: Nikolaj William Coster-Waldau (born 27 July 1970) is a Danish actor and producer. He graduated from the Danish National School of Performing Arts in Copenhagen in 1993, and had his breakthrough role in Denmark with the film Nightwatch (1994). He played Jaime Lannister in the HBO fantasy drama series Game of Thrones, for which he received two Primetime Emmy Award nominations for Outstanding Supporting Actor in a Drama Series. Coster-Waldau has appeared in numerous films in his native Denmark and Scandinavia, including Headhunters (2011) and A Thousand Times Good Night (2013). In the U.S, his debut film role was in the war film Black Hawk Down (2001), playing Medal of Honor recipient Gary Gordon. He then played a detective in the short-lived Fox television series New Amsterdam (2008), and appeared in the 2009 Fox television film Virtuality, originally intended as a pilot.
    Thought 2: Because he "appeared in the 2009 Fox television film Virtuality", he should have worked with the Fox Broadcasting Company.
    Action 2: finish("SUPPORTS")
    """,
    """
    Question: Stranger Things is set in Bloomington, Indiana.
    Thought 1: I should search for Stranger Things, and see if it is set in Bloomington, Indiana.
    Action 1: search("Stranger Things")
    Observation 1: Stranger Things is an American science fiction horror drama television series created by the Duffer Brothers. Set in the 1980s, primarily in the fictional town of Hawkins, Indiana, the series centers on a number of mysteries and supernatural events occurring around the town and their impact on an ensemble of child and adult characters.
    Thought 2: The observation says that it is set in a "fictional town of Hawkins, Indiana", so it is not set in Bloomington.
    Action 2: finish("REFUTES")
    """,
    """
    Question: Beautiful reached number two on the Billboard Hot 100 in 2003.
    Thought 1: I need to search the song Beautiful and find if it reached number two on the Billboard Hot 100 in 2003.
    Action 1: search("Beautiful")
    Observation 1: Could not find [Beautiful]. Similar: ['Beautiful', 'Beautiful, Beautiful', 'A Beautiful Mind (film)', 'Beautiful (Christina Aguilera song)', 'Life Is Beautiful'].
    Thought 2: From suggestions, I should search "Beautiful (Christina Aguilera song)" to find the song.
    Action 2: search("Beautiful (Christina Aguilera song)")
    Observation 2: "Beautiful" is a song recorded by American singer Christina Aguilera for her fourth studio album, Stripped (2002).
    Thought 3: It does not mention Billboard, so I need to look up "Billboard Hot 100" to find if it reached number two on it in 2003.
    Action 3: lookup("Billboard Hot 100")
    Observation 3: The song peaked at number two on the Billboard Hot 100 in the United States, where it was certified Gold for 500,000 units shipped.
    Thought 4: It only says the song peaked at number two on the Billboard Hot 100, but not if it was in 2003. I am not sure if this claim is true or not.
    Action 4: finish("NOT ENOUGH INFO")
    """
]


def config_agent(model_kwargs: Dict, examples: Optional[List[str]] = []) -> ReActAgent:
    """
    Configure the react agent

    Args:
        model_kwargs (Dict): a type of model
        examples (Optional[List[str]], optional): a list of examples. Defaults to [].
        tools (List[Union[Callable, AsyncCallable, FunctionTool]]): a list of tools for agent to make functional calls
    Returns:
        ReActAgent: the configured agent
    """
    
    preset_prompt_kwargs = {'examples': examples} if len(examples) else {}
    
    # set up tools
    tools = [FunctionTool.from_defaults(fn=search), FunctionTool.from_defaults(fn=lookup)]
    model_client = OpenAIClient if 'gpt' in model_kwargs.get('model', '') else GroqAPIClient
    
    return ReActAgent(
        tools=tools, max_steps=7, model_client=model_client,
        model_kwargs=model_kwargs, preset_prompt_kwargs=preset_prompt_kwargs,
        template=FEVER_REACT_AGENT_SYSTEM_PROMPT
    )


def run_query(agent: ReActAgent, question: str, gt_answer:str) -> Dict[str, float]:
    """
    Run queries and calculate the evaluation metrics
    """
    start_time = time.time()
    pred_answer = agent(question)
    pred_answer = normalize_answer(pred_answer)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Question: {question}, \ngt_answer: {gt_answer}, \npred_answer: {pred_answer}\n")

    em = EM_evaluator.compute_match_acc_single_query(pred_answer=pred_answer, gt_answer=gt_answer)
    fm = FM_evaluator.compute_match_acc_single_query(pred_answer=pred_answer, gt_answer=gt_answer)
    
    return {
        "EM": em,
        "FM": fm,
        "time": elapsed_time
    }
    
def experiment(num_questions: int, dataset: List[Dict[str, Any]], model_kwargs: Dict, examples: Optional[List[str]] = []) -> Dict[str, float]:
    """
    Perform react agent experiment, evaluation metrics are Exact Match and Fuzzy Match

    Args:
        num_questions (int): number of total evaluation records
        dataset (List[Dict[str, Any]]): the HotPotQA dataset
        model_kwargs (Dict): model configurations, e.g. {"model":, "gpt-3.5-turbo", "temperature": 0.0}
        examples (Optional[List[str]], optional): a list of examples. Defaults to [].

    Returns:
        Dict[str, float]: return the evaluations
    """
    
    logger.info(f"model_kwargs: {model_kwargs}")
    
    # Initialize the agent once if configuration does not need to change each iteration
    react_agent = config_agent(model_kwargs=model_kwargs, examples=examples)
    total_metrics = {"N": 0, "EM": 0, "FM": 0, "time": 0}
    for i in range(num_questions):
        question = dataset[i]["claim"]
        gt_answer = normalize_answer(dataset[i]["label"])
        
        result = run_query(react_agent, question, gt_answer)

        total_metrics["N"] += 1 # number of questions
        total_metrics["EM"] += result["EM"]
        total_metrics["FM"] += result["FM"]
        total_metrics["time"] += result["time"]

    # Calculate averages
    average_metrics = {key: val / num_questions for key, val in total_metrics.items()}
    average_metrics["N"] = num_questions
    logger.info(f"Average metrics: {average_metrics}")
    return average_metrics



# setup evaluators
EM_evaluator = AnswerMacthEvaluator(type="exact_match")
FM_evaluator = AnswerMacthEvaluator(type="fuzzy_match")

# load test data
file = open('./tests/benchmark/ReAct_agent/paper_data/paper_dev_10.json')
dataset = json.load(file)

# define the arguments, follow the paper's argument settings
gpt_3_turbo_model_kwargs = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.0,
        "max_tokens": 100,
        "top_p": 1,
        "frequency_penalty":0.0,
        "presence_penalty":0.0,
    }

gpt_4o_model_kwargs = {
        "model": "gpt-4o",
        "temperature": 0.0,
        "max_tokens": 100,
        "top_p": 1,
        "frequency_penalty":0.0,
        "presence_penalty":0.0,
    }
gpt_4_turbo_model_kwargs = {
        "model": "gpt-4-turbo-preview",
        "temperature": 0.0,
        "max_tokens": 100,
        "top_p": 1,
        "frequency_penalty":0.0,
        "presence_penalty":0.0,
    }

llama3_model_kwargs = {
        "model": "llama3-70b-8192",  # llama3 is not good with string formatting, llama3 8b is also bad at following instruction, 70b is better but still not as good as gpt-3.5-turbo
        "temperature": 0.0,
    }

num_questions = 10
# gpt_3_5_zero_shot = experiment(num_questions=num_questions, dataset=dataset, model_kwargs=gpt_3_turbo_model_kwargs)
# gpt_3_5_3_shot = experiment(num_questions=num_questions, dataset=dataset, model_kwargs=gpt_3_turbo_model_kwargs, examples=examples)
# gpt_3_5_6_shot = experiment(num_questions=num_questions, dataset=dataset, model_kwargs=gpt_3_turbo_model_kwargs, examples=examples)

gpt_4o_zero_shot = experiment(num_questions=num_questions, dataset=dataset, model_kwargs=gpt_4o_model_kwargs)
gpt_4o_3_shot = experiment(num_questions=num_questions, dataset=dataset, model_kwargs=gpt_4o_model_kwargs, examples=examples)

# gpt_4_turbo_zero_shot = experiment(num_questions=num_questions, dataset=dataset, model_kwargs=gpt_4_turbo_model_kwargs)
# gpt_4_turbo_3_shot = experiment(num_questions=num_questions, dataset=dataset, model_kwargs=gpt_4_turbo_model_kwargs, examples=examples)

# llama3_zero_shot = experiment(num_questions=num_questions, dataset=dataset, model_kwargs=llama3_model_kwargs)
# llama3_3_shot = experiment(num_questions=num_questions, dataset=dataset,  model_kwargs=llama3_model_kwargs, examples=examples)

# print(f"gpt_3_5_zero_shot: {gpt_3_5_zero_shot}")    
# print(f"gpt_3_5_3_shot: {gpt_3_5_3_shot}")    
# print(f"gpt_3_5_6_shot: {gpt_3_5_6_shot}")  
print(f"gpt_4o_zero_shot: {gpt_4o_zero_shot}")
print(f"gpt_4o_3_shot: {gpt_4o_3_shot}")
# print(f"gpt_4_turbo_zero_shot: {gpt_4_turbo_zero_shot}")
# print(f"gpt_4_turbo_3_shot: {gpt_4_turbo_3_shot}")
# print(f"llama3_zero_shot: {llama3_zero_shot}")    
# print(f"llama3_3_shot: {llama3_3_shot}")    

  

"""
NOTE: llama3 time might not accurate because it has request limit error
gpt_3_5_zero_shot: {'N': 10, 'EM': 0.0, 'FM': 0.0, 'time': 12.529078555107116}
gpt_3_5_3_shot: {'N': 10, 'EM': 0.3, 'FM': 0.4, 'time': 7.47766683101654}
gpt_4o_zero_shot: {'N': 10, 'EM': 0.7, 'FM': 0.7, 'time': 5.547899603843689}
gpt_4_turbo_zero_shot: {'N': 10, 'EM': 0.6, 'FM': 0.6, 'time': 14.762795424461364}
gpt_4o_3_shot: {'N': 10, 'EM': 0.7, 'FM': 0.7, 'time': 5.411731863021851}
gpt_4_turbo_3_shot: {'N': 10, 'EM': 0.7, 'FM': 0.7, 'time': 11.741541314125062}
llama3_zero_shot: {'N': 10, 'EM': 0.4, 'FM': 0.4, 'time': 45.428342413902286}
llama3_3_shot: {'N': 10, 'EM': 0.6, 'FM': 0.6, 'time': 42.95352940559387}

paper:
gpt3.5
zero_shot: 0 10 0.0 11.41517460346222
3 shot: 3 10 0.3 3.8097436904907225
4o few shot: 6 10 0.6 8.16726279258728
4o zero shot: 0 10 0.0 35.91357259750366
4-turbo 3 shot: 8 10 0.8 7.756631708145141
4 turbo zero shot: 0 10 0.0 36.0502730846405

"""