
# Agent is not a model or LLM model.
# Agent is better defined as a system that uses LLM models to plan and replan steps that each involves the usage of various tools,
# such as function calls, another LLM model based on the context and history (memory) to complete a task autonomously.


# REact agent can be useful for
# - Multi-hop reasoning [Q&A], including dividing the query into subqueries and answering them one by one.
# - Plan the usage of the given tools: highly flexible. Retriever, Generator modules or any other functions can all be wrapped as tools.

# The initial ReAct paper does not support different types of tools. We have greatly extended the flexibility of tool adaption, even including an llm tool
# to answer questions that cant be answered or better be answered by llm using its world knowledge.
# - Every react agent can be given a different tasks, different tools, and different LLM models to complete the task.
# - 'finish' tool is defined to finish the task by joining all subqueries answers.

# Reference:
# [1] LLM Agent survey: https://github.com/Paitesanshi/LLM-Agent-Survey
Agent is not a model or LLM model.

Agent is better defined as a system that uses LLM models to plan and replan steps that each involves the usage of various tools,
such as function calls, another LLM model based on the context and history (memory) to complete a task autonomously.

The future: the agent can write your prompts too. Check out dspy: https://github.com/stanfordnlp/dspy

In this directory, we add the general design patterns of agent, here are four (Thanks to Andrew Ng):

1️⃣ Reflection

- Self-Refine: Iterative Refinement with Self-Feedback
- Reflexion: Language Agents with Verbal Reinforcement Learning

2️⃣ Tool use

- Gorilla: Large Language Model Connected with Massive APIs
- MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action

3️⃣ Planning

- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face
- React

4️⃣ Multi-agent collaboration

- Communicative Agents for Software Development
- AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
