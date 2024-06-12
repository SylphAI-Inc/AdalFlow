ReAct Agent
=================

The goal of this tutorial is to:

1. Demonstrate how ``LightRAG`` implements the ReAct agent
2. Provide a Deep Dive on ReAct Agent(Reference)

What is an agent and why you need it?
------------------------------------------------

An agent, is better defined as a system that strategically uses LLM models and various tools to plan and execute steps.
Although LLMs and RAGs can generate text response with conversation history and internal knowledge,
they are unable to plan sequentially and decide which resource to use.

Introduction
-----------------------
Before explaining ``LightRAG Agent`` implementation, here is a quick introduction of ReAct Agent.

To solve a query, the `ReAct Agent <https://arxiv.org/pdf/2210.03629>`_, like its name(``Re``- Reason; ``Act`` - Act), 
first uses LLM to analyze the context and plan actions to answer the query(reasoning).
Then it takes actions to utilize external resources(action). For more details, please see the :ref:`deep-dive`.

LightRAG's Implementation
-----------------------------------------------------
Next, let's look at how ``LightRAG`` makes the implementation convenient. In ``LightRAG``, the ReAct agent is a type of :ref:`generator` that runs multiple sequential steps to generate the final response, with designed prompt, external functions(named as ``tools``) and ``JsonParser output_processors``.

1. **Prompt:** We have a easy-to-customizable prompt template designed for ReAct agent that takes in the tools, examples, and context(step history), etc. :ref:`Prompt <DEFAULT_REACT_AGENT_SYSTEM_PROMPT>`.

2. **Tools:** ReAct Agent needs to plan the tool to use, which means it needs to access the tools' descriptions. 
``LightRAG`` provides dynamic tool handling, using ``FunctionTool`` to encapsulate tool functionalities. The metadata(function name, description, and parameters) will be extracted and passed to the prompt automatically. This process not only makes tool integration more seamless but also enhances developer efficiency by allowing straightforward definition and management of tools.

Here is the example to illustrate the usage of ``FunctionTool``. It's easy to set up using ``from_defaults``.

.. code-block:: python

    from lightrag.core.tool_helper import FunctionTool

    # define the tools
    def multiply(a: int, b: int) -> int:
        '''Multiply two numbers.'''
        return a * b
    def add(a: int, b: int) -> int:
        '''Add two numbers.'''
        return a + b

    tools = [
                FunctionTool.from_defaults(fn=multiply),
                FunctionTool.from_defaults(fn=add),
            ]

    for tool in tools:
        name = tool.metadata.name
        description = tool.metadata.description
        parameter = tool.metadata.fn_schema_str
        print(f"Function name: {name}")
        print(f"Function description: {description}")
        print(f"Function parameter: {parameter}")

    # Function name: multiply
    # Function description: multiply(a: int, b: int) -> int
    # Multiply two numbers.
    # Function parameter: {"type": "object", "properties": {"a": {"type": "int"}, "b": {"type": "int"}}, "required": ["a", "b"]}
    # Function name: add
    # Function description: add(a: int, b: int) -> int
    # Add two numbers.
    # Function parameter: {"type": "object", "properties": {"a": {"type": "int"}, "b": {"type": "int"}}, "required": ["a", "b"]}

The agent will then call these external functions based on the function descriptions.
In addition to user-defined tools, the :class:`ReActAgent <components.agent.react_agent.ReActAgent>` built-in ``llm_tool``
for leveraging LLM's internal knowledge, and ``finish`` for completing processes. Developers have the flexibility to enable or disable these as needed.

3. **Output Parser:** ``LightRAG`` requests the model to output intermediate Thought and Action as JSON, which facilitates better error handling and easier data manipulation than strings. For example,
    
.. code-block:: json
    
    {
        "thought": "<Why you are taking this action>",
        "action": "ToolName(<args>, <kwargs>)"
    }

This format allows the ``LightRAG`` JSON parser to efficiently decode the model's output and extract arguments. 
The parsed data is then utilized by the ``StepOutput`` class to manage the flow of thought, action and observation.

4. **Example:** Let's see a Q&A agent example:

.. code-block:: python

    from lightrag.core.tool_helper import FunctionTool
    from lightrag.components.agent.react_agent import ReActAgent
    from lightrag.components.model_client import OpenAIClient
    from lightrag.components.model_client import GroqAPIClient

    import dotenv
    # load evironment
    dotenv.load_dotenv(dotenv_path=".env", override=True)

    # define the tools
    def multiply(a: int, b: int) -> int:
        '''Multiply two numbers.'''
        return a * b
    def add(a: int, b: int) -> int:
        '''Add two numbers.'''
        return a + b

    tools = [
            FunctionTool.from_defaults(fn=multiply),
            FunctionTool.from_defaults(fn=add),
        ]

    # for tool in tools:
    #    name = tool.metadata.name
    #    description = tool.metadata.description
    #    parameter = tool.metadata.fn_schema_str
    #    print(f"Function name: {name}")
    #    print(f"Function description: {description}")
    #    print(f"Function parameter: {parameter}")
        
        
    examples = [
            """
            User: What is 9 - 3?
            You: {
                "thought": "I need to subtract 3 from 9, but there is no subtraction tool, so I ask llm_tool to answer the query.",
                "action": "llm_tool('What is 9 - 3?')"
            }
            """
    ]

    preset_prompt_kwargs = {"example": examples}
    llm_model_kwargs = {
        "model": "llama3-70b-8192",
        "temperature": 0.0
    }

    agent = ReActAgent(
        tools=tools,
        model_client=GroqAPIClient(),
        model_kwargs=llm_model_kwargs,
        max_steps=3,
        preset_prompt_kwargs=preset_prompt_kwargs
        )

    import time        
    queries = ["What is 3 add 4?", "3*9=?"]
    average_time = 0
    for query in queries:
        t0 = time.time()
        answer = agent(query)
    
    # Answer: The answer is 7.
    # Answer: The answer is 27.

5. **Subquery and History:** Moreover, in our design, the agent will potentially divide a query into subqueries, join all subqueries answers and finish the task. Developers can customize the prompt depending on the use cases.
The intermediate step history is managed. The agent will visit its previous reasoning, action and observations before making decisions.

.. _deep-dive:

ReAct Agent Deep Dive
---------------------------
Please read this section if you need more information on ReAct agent.

`ReAct Agent <https://arxiv.org/pdf/2210.03629>`_, like its name(``Re``- Reason; ``Act`` - Act), is a framework generating reasoning and taking actions in an interleaved manner. The reasoning step guides the model to action plans and the action step allows the agent to interact with external sources such as knowledge bases. 

The paper shows:
1. ReAct with few-shot prompt and Wikipedia API interaction outperforms chain-of-thought on `HotpotQA <https://arxiv.org/pdf/1809.09600>`_ (Question and Answering) and `Fever <https://arxiv.org/pdf/1803.05355v3>`_ (Fact Verification).
2. ReAct performs well on two interactive decision making benchmarks.

**1. Overall Workflow**

Unlike the reasoning only and acting only approaches, given a query, the ReAct agent will go through a sequence of steps to solve the problem. (`Source <https://react-lm.github.io/>`_)

Here is an example from the paper that demonstrates the workflow.

.. image:: ../../../images/ReAct.jpg

The environment contains user query, step histories, observations, and external sources.

At each step, the agent:

- **[Thought]** In response to the environment and user query, the agent uses its LLM to generate a strategic thought that outlines a plan or hypothesis guiding the subsequent action. 

- **[Action]** The agent executes the action.

The environment will be updated:

- **[Observation]** The observation is created after the action is done.

Then the agent iteratively generates thoughts based on latest observation and context(previous steps), takes actions and gets new observations. 

The termination condition is: 

* The agent finds the answer and takes "finish" action.

* The agent fails to get the answer when the defined max steps is reached. Return nothing.

**2. Action Space**

Now we understand the 3 different stages: Thought, Action, Observation. Let's focus on Action, one of agents' uniqueness. 

Actions refer to the tools the agent uses to interact with the environment and creates observations.
Note: the paper defines Thought(or reasoning trace) as a *language level action* but it is not included in the action space because it doesn't impact the environment. 

Use ``HotpotQA`` dataset as an example, what external source do we need to answer questions?

`HotpotQA <https://arxiv.org/pdf/1809.09600>`_ contains Wikipedia-based questions that require multi-hop reasoning. Therefore, the agent will need to query the Wikipedia API.

In the `ReAct paper <https://arxiv.org/pdf/2210.03629>`_, researchers include 3 actions in the "action space" (simplified version here):

* search[entity], returns the first 5 sentences from the corresponding entity wiki page if it exists, or else suggests top-5 similar entities. 

* lookup[string], simulating Ctrl+F functionality on the browser. 

* finish[answer], which would finish the current task with answer. 

**3. Components**

With the workflow and action space, next, let's focus on the components needed to implement the agent.

* **prompt:** Besides the role and task-specific description, the key in ReAct prompting is to define the tools to use in the prompt.

* **function call:** In the implementation, each action is essentially a function to call. Clear functionality definition is important for the agent to determine which action to take next.

* **parser:** The agent is built on LLMs. It takes in the prompt with context, generates thought and determine the action to take in text response. 
To really call functions, we need to parse the text response to get the parameters for the determined function.