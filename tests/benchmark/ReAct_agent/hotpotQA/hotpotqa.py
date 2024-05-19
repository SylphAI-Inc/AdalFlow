"""
https://arxiv.org/abs/2210.03629, published in Mar, 2023

Use LightRAG agent component to build the same version of ReAct agent as in the paper.
Apply the API code from the Paper (open-source).
Use the default prompt without any example.
"""


import dotenv
from tests.benchmark.ReAct_agent.paper_code import wikienv, wrappers
from components.api_client.openai_client import OpenAIClient
from components.agent.react_agent import ReActAgent
from core.tool_helper import FunctionTool
import time


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

class HotpotQAReActAgent:
    """
    Use LightRAG agent component to build the same version of ReAct agent as in the paper.
    Apply the API code from the Paper (open-source).
    Use the default prompt without any example.
    """
    def __init__(self, model="gpt-3.5-turbo"):
        dotenv.load_dotenv(dotenv_path=".env", override=True)
        self.env = self.setup_environment()
        self.agent = self.setup_agent(model)

    def setup_environment(self):
        """
        Sets up the Wiki environment and wraps it with necessary wrappers.
        """
        env = wikienv.WikiEnv()
        env = wrappers.HotPotQAWrapper(env, split="dev")
        env = wrappers.LoggingWrapper(env)
        return env

    def setup_agent(self, model):
        """
        Configures and returns a ReAct Agent with specified tools and model.
        """
        tools = [
            FunctionTool.from_defaults(fn=self.search),
            FunctionTool.from_defaults(fn=self.lookup)
        ]
        return ReActAgent(
            tools=tools,
            model_client=OpenAIClient,  # Ensure OpenAIClient is instantiated
            model_kwargs={"model": model}
        )

    def search(self, entity: str) -> str:
        """
        Searches for an entity on Wikipedia.
        """
        self.env.step(f"search({entity})")
        return self.env.obs

    def lookup(self, keyword: str) -> str:
        """
        Searches within a specific text passage and returns the next sentence containing the keyword.
        """
        self.env.step(f"lookup({keyword})")
        return self.env.obs


# Usage Example
if __name__ == "__main__":
    react_agent = HotpotQAReActAgent(model="gpt-3.5-turbo")
    
    # sample 
    average_time = 0
    num_questions = 10
    for i in range(num_questions):
        question = react_agent.env.reset(idx=i)  # get the question
        t0 = time.time()
        answer = react_agent.agent(question)
        average_time += time.time() - t0
        print(f"Answer: {answer}")
    
    print(f"Average time: {average_time / num_questions}")
    """
    Results: 11s per query, gpt-3.5-turbo, without setting max step
    
    """