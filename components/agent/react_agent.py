"""
https://arxiv.org/abs/2210.03629, published in Mar, 2023

Agent is not a model or LLM model.
Agent is better defined as a system that uses LLM models to plan and replan steps that each involves the usage of various tools,
such as function calls, another LLM model based on the context and history (memory) to complete a task autonomously.

The future: the agent can write your prompts too. Check out dspy: https://github.com/stanfordnlp/dspy

REact agent can be useful for
- Multi-hop reasoning [Q&A], including dividing the query into subqueries and answering them one by one.
- Plan the usage of the given tools: highly flexible. Retriever, Generator modules or any other functions can all be wrapped as tools.

The initial ReAct paper does not support different types of tools. We have greatly extended the flexibility of tool adaption, even including an llm tool
to answer questions that cant be answered or better be answered by llm using its world knowledge.
- Every react agent can be given a different tasks, different tools, and different LLM models to complete the task.
- 'finish' tool is defined to finish the task by joining all subqueries answers.
"""

from typing import List, Union, Callable, Optional, Any, Dict
from dataclasses import dataclass


from core.generator import Generator
from core.component import Component
from core.prompt_builder import Prompt
from core.tool_helper import FunctionTool, AsyncCallable
from core.string_parser import JsonParser, parse_function_call


DEFAULT_REACT_AGENT_PROMPT = r"""
<<SYS>>
{# role/task description #}
You task is to answer user's query with minimum steps and maximum accuracy using the tools provided.
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
Learn from the examples:
{% for example in examples %}
{{ example }}
{% endfor %}
{% endif %}
<</SYS>>
-----------------
User query: {{user_query}}
{# History #}
{% for history in step_history %}
Step {{history.step}}:
{
 "thought": "{{history.thought}}",
 "action": "{{history.action}}",
}
"observation": "{{history.observation}}"
{% endfor %}
You:
"""

# NOTE: if the positional and keyword arguments are not working well,
# you can let it be a json string and use only keyword arguments and use json parser to parse the arguments instead of parse_function_call


@dataclass
class StepOutput:
    step: int
    thought: str
    action: str
    fun_name: Optional[str] = None  # parsed from action
    fun_args: Optional[List[Any]] = None  # parsed from action
    fun_kwargs: Optional[Dict[str, Any]] = None  # parsed from action
    observation: Optional[str] = (
        None  # when step is created, observation is not available, the funtion result
    )

    def __str__(self):
        return f"Thought {self.step}: {self.thought}\nAction {self.step}: {self.action}\nObservation {self.step}: {self.observation}"


class ReActAgent(Generator):
    r"""
    ReActAgent is just a subclass of Generator with more functionalities to plan and replan steps that each involves the usage of various tools,
    """

    def __init__(
        self,
        *,
        # added arguments specifc to React
        examples: List[str] = [],
        tools: List[Union[Callable, AsyncCallable, FunctionTool]] = [],
        max_steps: int = 10,
        # the following arguments are inherited from Generator
        prompt: Prompt = Prompt(DEFAULT_REACT_AGENT_PROMPT),  # reset the default prompt
        output_processors: Optional[
            Component
        ] = JsonParser(),  # reset the default output_processors
        **other_generator_kwargs,
    ):
        super().__init__(
            prompt=prompt, output_processors=output_processors, **other_generator_kwargs
        )
        self.examples = examples
        self.tools = tools
        self.max_steps = max_steps

        self.additional_llm_tool = Generator(
            **other_generator_kwargs
        )  # use any other setting and set prompt and output_processors to default

        def llm_tool(input: str) -> str:
            """
            I answer any input query with llm's world knowledge. Use me as a fallback tool or when the query is simple.
            """
            # use the generator to answer the query
            try:
                return self.additional_llm_tool(input=input)
            except Exception as e:
                print(f"Error using the generator: {e}")

            return None

        def finish(answer: str) -> str:
            """
            Finish the task by joinging all subqueries answers.
            """
            return answer

        self.tools.extend([llm_tool, finish])
        # convert all functions to FunctionTool, and track how to call each function, either call or acall
        self.tools = [
            (
                tool
                if isinstance(tool, FunctionTool)
                else FunctionTool.from_defaults(fn=tool)
            )
            for tool in self.tools
        ]

        self.tools_map = {tool.metadata.name: tool for tool in self.tools}
        self.step_history: List[StepOutput] = []
        self.output_processors = output_processors

    def reset(self):
        self.step_history = []

    def _parse_text_response(
        self, json_obj_response: Dict[str, Any], step: int
    ) -> Optional[StepOutput]:
        """
        Parse the json output
        """
        try:
            # json_obj_response = self.output_processors(response)
            thought_key = "thought"
            action_key = "action"
            thought = json_obj_response.get(thought_key, "")
            action = json_obj_response.get(action_key, "")
            return StepOutput(step=step, thought=thought, action=action)
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None

    def _execute_action(self, action_step: StepOutput) -> Optional[StepOutput]:
        """
        Parse the action string to a function call and execute it. Update the action_step with the result.
        """
        action = action_step.action
        try:
            fun_name, args, kwargs = parse_function_call(action, self.tools_map)
            print(f"fun_name: {fun_name}, args: {args}, kwargs: {kwargs}")
            fun: Union[Callable, AsyncCallable] = self.tools_map[fun_name].fn
            result = fun(*args, **kwargs)
            action_step.fun_name = fun_name
            action_step.fun_args = args
            action_step.fun_kwargs = kwargs

            action_step.observation = result
            return action_step
        except Exception as e:
            print(f"Error executing {action}: {e}")
            # pass the error as observation so that the agent can continue and correct the error in the next step
            action_step.observation = f"Error executing {action}: {e}"
            return action_step

    def _run_one_step(self, input: str, step: int) -> str:
        """
        Run one step of the agent.
        """
        prompt_kwargs = {
            "user_query": input,
            "tools": self.tools,
            "step_history": self.step_history,
            "examples": self.examples,
        }
        # call the super class to generate the response
        response = super().call(
            input=input, prompt_kwargs=prompt_kwargs, model_kwargs=self.model_kwargs
        )
        parsed_response = self._parse_text_response(
            json_obj_response=response, step=step
        )
        # execute the action
        if parsed_response and parsed_response.action:
            parsed_response = self._execute_action(parsed_response)
        else:
            print(f"Failed to parse response for step {step}")
        self.step_history.append(parsed_response)

        return response

    def call(self, input: str) -> str:
        """
        Run the agent on the given input.
        """
        for i in range(self.max_steps):
            step = i + 1
            try:
                self._run_one_step(input, step)
                if (
                    self.step_history[-1].fun_name
                    and self.step_history[-1].fun_name == "finish"
                ):
                    break
            except Exception as e:
                error_message = f"Error running step {step}: {e}"
                print(error_message)

        answer = self.step_history[-1].observation
        print(f"step_history: {self.step_history}")
        self.reset()
        return answer


if __name__ == "__main__":
    from components.api_client.groq_client import GroqAPIClient
    import utils.setup_env

    def multiply(a: int, b: int) -> int:
        """
        Multiply two numbers.
        """
        return a * b

    def add(a: int, b: int) -> int:
        """
        Add two numbers.
        """
        return a + b

    def search(query: str) -> str:
        """
        Search the web for the given query.
        """
        return "python programming is a great way to learn programming"

    tools = [
        FunctionTool.from_defaults(fn=multiply),
        FunctionTool.from_defaults(fn=add),
        # FunctionTool.from_defaults(fn=search),
    ]
    llm_model_kwargs = {
        "model": "llama3-70b-8192",  # llama3 is not good with string formatting, llama3 8b is also bad at following instruction, 70b is better but still not as good as gpt-3.5-turbo
        # mistral also not good: mixtral-8x7b-32768, but with better prompt, it can still work
        "temperature": 0.0,
    }

    # planner = Generator(model_client=GroqAPIClient(), model_kwargs=llm_model_kwargs)
    # settings = {
    #     "provider": "openai",
    #     "model": "gpt-3.5-turbo",
    #     "temperature": 0.0,
    # }
    # planner = OpenAIGenerator(**settings)
    examples = [
        # r"""
        # User: What is 9 - 3?
        # You: {
        #     "thought": "I need to subtract 3 from 9, but there is no subtraction tool, so I ask llm_tool to answer the query.",
        #     "action": "llm_tool('What is 9 - 3?')"
        # }
        # """
    ]
    agent = ReActAgent(
        examples=examples,
        tools=tools,
        max_steps=10,
        model_client=GroqAPIClient(),
        model_kwargs=llm_model_kwargs,
    )
    print(agent)
    # agent = ReActAgent(generator=planner, tools=tools, max_steps=10, examples=examples)
    queries = [
        # "What is 2 times 3?",
        # "What is 3 plus 4?",
        # "What is the capital of France? and what is 4 times 5 then add 3?",  # this is actually two queries, or a multi-hop query
        "Li adapted her pet Apple in 2017 when Apple was only 2 months old, now we are at year 2024, how old is Li's pet Apple?",
    ]
    """
    Results: mixtral-8x7b-32768, 0.9s per query
    llama3-70b-8192, 1.8s per query
    gpt-3.5-turbo, 2.2s per query
    """
    import time

    average_time = 0
    for query in queries:
        t0 = time.time()
        answer = agent(query)
        average_time += time.time() - t0
        print(f"Answer: {answer}")
    print(f"Average time: {average_time / len(queries)}")

    # test multiply(2024 - 2017, 12)
