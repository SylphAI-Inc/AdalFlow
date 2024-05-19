import dotenv
from datasets import load_dataset

from tests.benchmark.hotpotQA.paper_code import wikienv
from tests.benchmark.hotpotQA.paper_code import wrappers
from components.api_client.openai_client import OpenAIClient
from components.agent.react_agent import ReActAgent
from core.tool_helper import FunctionTool
import json
import time

dotenv.load_dotenv(dotenv_path=".env", override=True)

# Load the dataset and select the first 10 as the showcase
# More info about the HotpotQA dataset can be found at https://huggingface.co/datasets/hotpot_qa
dataset = load_dataset(path="hotpot_qa", name="fullwiki")
prompt_file = './tests/benchmark/prompts_naive.json'
with open(prompt_file, 'r') as f:
    prompt_dict = json.load(f)
webthink_examples = prompt_dict['webthink_simple6']

instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""


# set up the env
env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)


import requests
from bs4 import BeautifulSoup

def search(entity):
    """
    Searches for an entity on Wikipedia.
    """
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity.replace(' ', '+')}"
    response_text = requests.get(search_url).text
    soup = BeautifulSoup(response_text, 'html.parser')
    
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    if result_divs:
        results = [div.get_text().strip() for div in result_divs][:5]
        return f"Could not find {entity}. Similar: {results}."
    else:
        return "No results found."


def lookup(keyword):
    """
    searches within a specific text passage and returns the next sentence containing the keyword.
    """
    env.step(f"lookup[{keyword}]")  # Lookup the keyword 'interpreted'
    return env.obs    

tools = [
    FunctionTool.from_defaults(fn=search),
    FunctionTool.from_defaults(fn=lookup),
    # FunctionTool.from_defaults(fn=finish),
]
model_kwargs = {"model": "gpt-3.5-turbo"}


preset_prompt_kwargs = {"examples": [webthink_examples]}
react_agent = ReActAgent(
    tools=tools,
    model_client=OpenAIClient,
    model_kwargs=model_kwargs,
    # max_steps=10,
    preset_prompt_kwargs=preset_prompt_kwargs
    )

average_time = 0
num_questions = 1
for i in range(num_questions):
    question = env.reset(idx=i) # get the question
    t0 = time.time()
    answer = react_agent(question)
    average_time += time.time() - t0
    print(f"Answer: {answer}")
print(f"Average time: {average_time / num_questions}")


# queries = ["Question: What movie did actress Irene Jacob complete before the American action crime thriller film directed by Stuart Bird?"]
# import time

# print(webthink_examples)

# average_time = 0
# for i, query in enumerate(queries):
#     env.reset(idx=i)      # Reset the environment to the initial state   
#     t0 = time.time()
#     answer = react_agent(query)
#     average_time += time.time() - t0
#     print(f"Answer: {answer}")
# print(f"Average time: {average_time / len(queries)}")