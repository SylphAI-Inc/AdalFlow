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
from benchmarks.tests.ReAct_agent.hotpotQA.tools import search, lookup, normalize_answer
from eval.evaluators import AnswerMacthEvaluator
import logging
import json
from typing import List, Union, Callable, Optional, Any, Dict
from core.tool_helper import FunctionTool, AsyncCallable


logger = logging.getLogger(__name__)
logging.basicConfig(filename='./logs/hotpot.log', level=logging.INFO)

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
examples = [
"""Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action 1: search("Colorado orogeny")
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.
Action 2: lookup("eastern sector")
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action 3: search("High Plains")
Observation 3: High Plains refers to one of two distinct land regions:
Thought 4: I need to instead search High Plains (United States).
Action 4: search("High Plains (United States)")
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action 5: finish("1,800 to 7,000 ft")""",
"""Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: search("Milhouse")
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: lookup("named after")
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: finish("Richard Nixon")""",
"""Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.
Action 1: search("Adam Clayton Powell")
Observation 1: Could not find ["Adam Clayton Powell"]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].
Thought 2: To find the documentary, I can search Adam Clayton Powell (film).
Action 2: search("Adam Clayton Powell (film)")
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.
The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.
Thought 3: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action 3: finish("The Saimaa Gesture")""",
"""Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action 1: search("Nicholas Ray")
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action 2: search("Elia Kazan")
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action 3: finish("director, screenwriter, actor")""",
"""Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: search("Arthur's Magazine")
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. 
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action 2: search("First for Women")
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. 
Thought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action 3: finish("Arthur's Magazine")""",
"""Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action 1: search("Pavel Urysohn")
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.
Action 2: search("Leonid Levin")
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. 
Thought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. 
Action 3: finish("yes")"""
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
        model_kwargs=model_kwargs, preset_prompt_kwargs=preset_prompt_kwargs
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
        question = dataset[i]["question"]
        gt_answer = normalize_answer(dataset[i]["answer"])
        
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
file = open('./tests/benchmark/ReAct_agent/paper_data/hotpot_dev_v1_simplified_random_100.json')
dataset = json.load(file)

# define the arguments, follow the paper's argument settings
gpt_model_kwargs = {
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

num_questions = 5
# gpt_3_5_zero_shot = experiment(num_questions=num_questions, dataset=dataset, model_kwargs=gpt_model_kwargs)
# gpt_3_5_3_shot = experiment(num_questions=num_questions, dataset=dataset, model_kwargs=gpt_model_kwargs, examples=examples[:3])
# gpt_3_5_6_shot = experiment(num_questions=num_questions, dataset=dataset, model_kwargs=gpt_model_kwargs, examples=examples)

# gpt_4o_zero_shot = experiment(num_questions=num_questions, dataset=dataset, model_kwargs=gpt_4o_model_kwargs)
# gpt_4o_6_shot = experiment(num_questions=num_questions, dataset=dataset, model_kwargs=gpt_4o_model_kwargs, examples=examples)

# gpt_4_turbo_zero_shot = experiment(num_questions=num_questions, dataset=dataset, model_kwargs=gpt_4_turbo_model_kwargs)
gpt_4_turbo_6_shot = experiment(num_questions=num_questions, dataset=dataset, model_kwargs=gpt_4_turbo_model_kwargs, examples=examples)

# llama3_zero_shot = experiment(num_questions=num_questions, dataset=dataset, model_kwargs=llama3_model_kwargs)
# llama3_6_shot = experiment(num_questions=num_questions, dataset=dataset,  model_kwargs=llama3_model_kwargs, examples=examples)

# print(f"gpt_3_5_zero_shot: {gpt_3_5_zero_shot}")    
# print(f"gpt_3_5_3_shot: {gpt_3_5_3_shot}")    
# print(f"gpt_3_5_6_shot: {gpt_3_5_6_shot}")  
# print(f"gpt_4o_zero_shot: {gpt_4o_zero_shot}")
# print(f"gpt_4o_6_shot: {gpt_4o_6_shot}")
# print(f"gpt_4_turbo_zero_shot: {gpt_4_turbo_zero_shot}")
print(f"gpt_4_turbo_6_shot: {gpt_4_turbo_6_shot}")
# print(f"llama3_zero_shot: {llama3_zero_shot}")    
# print(f"llama3_6_shot: {llama3_6_shot}")    

  

"""
NOTE: llama3 time might not accurate because it has request limit error

first 10 records in the paper's dev dataset(7400+):
gpt_3_5_zero_shot: {'EM': 0.0, 'FM': 0.5, 'time': 46.83056600093842, 'Average step': 6.1}
gpt_3_5_6_shot: {'EM': 0.0, 'FM': 0.2, 'time': 12.900165343284607, 'Average step': 6.0}
llama3_zero_shot: {'EM': 0.0, 'FM': 0.4, 'time': 26.216207814216613, 'Average step': 6.1}
llama3_6_shot: {'EM': 0.1, 'FM': 0.5, 'time': 18.405735325813293, 'Average step': 7.1}

first 10 questions in the randomly selected 100 questions
gpt_3_5_zero_shot: {'EM': 0.0, 'FM': 0.3, 'time': 13.242103695869446, 'Average step': 6.1}
gpt_3_5_6_shot: {'EM': 0.0, 'FM': 0.4, 'time': 11.547260642051697, 'Average step': 5.4}

llama3_zero_shot: {'EM': 0.2, 'FM': 0.4, 'time': 137.26454865932465, 'Average step': 6}
llama3_6_shot: {'EM': 0.4, 'FM': 0.6, 'time': 223.38610920906066, 'Average step': 5.6}

gpt_4o_zero_shot: {'EM': 0.0, 'FM': 0.7, 'time': 8.016173100471496, 'Average step': 3.2}
gpt_4o_6_shot: {'EM': 0.4, 'FM': 0.7, 'time': 9, 'Average step': 3.7}

gpt_4_turbo_zero_shot: {'EM': 0.3, 'FM': 0.8, 'time': 11.181010842323303, 'Average step': 3.4}
gpt_4_turbo_6_shot: {'EM': 0.5, 'FM': 0.6, 'time': 11.961152362823487, 'Average step': 3.5}

randomly selected 100 records
gpt_3_5_zero_shot: {'EM': 0.02, 'FM': 0.23, 'time': 16.584252796173097, 'Average step': 5.93}
gpt_3_5_6_shot: {'EM': 0.02, 'FM': 0.09, 'time': 10.081220099925995, 'Average step': 6.78}
"""