"""
This tutorial demonstrates how to use MultiRunner to coordinate multiple agents in AdalFlow.

Key concepts covered:
1. Creating multiple agents with different roles
2. Using MultiRunner to coordinate agents
3. Handling inter-agent communication
4. Configuring parallel execution
"""

import logging
from typing import Dict, Any
from adalflow.core.agent import Agent
from adalflow.core.multirunner import MultiRunner
from adalflow.core.types import FunctionTool

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Example function tools for demonstration
def research(topic: str) -> str:
    """Research a given topic."""
    return f"Research results for {topic}"

def summarize(text: str) -> str:
    """Summarize a given text."""
    return f"Summary of {text[:50]}..."

def main():
    # 1. Create research agent
    research_tools = [
        FunctionTool(
            fn=research,
            name="research",
            description="Research a given topic",
        )
    ]

    research_agent_config = {
        "name": "research_agent",
        "model_client": {
            "component_name": "OpenAIClient",
            "component_config": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.7
            }
        },
        "model_kwargs": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7
        },
        "template": "You are a research assistant. Use the tools provided to gather information.",
        "tools": research_tools,
        "output_type": str
    }

    research_agent = Agent.from_config(research_agent_config)

    # 2. Create summary agent
    summary_tools = [
        FunctionTool(
            fn=summarize,
            name="summarize",
            description="Summarize a given text",
        )
    ]

    summary_agent_config = {
        "name": "summary_agent",
        "model_client": {
            "component_name": "OpenAIClient",
            "component_config": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.5
            }
        },
        "model_kwargs": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.5
        },
        "template": "You are a summary assistant. Use the tools provided to create concise summaries.",
        "tools": summary_tools,
        "output_type": str
    }

    summary_agent = Agent.from_config(summary_agent_config)

    # 3. Create MultiRunner configuration
    multirunner_config = {
        "agents": {
            "research_agent": research_agent,
            "summary_agent": summary_agent
        },
        "max_steps": 3,
        "output_type": str
    }

    # 4. Create MultiRunner
    multirunner = MultiRunner.from_config(multirunner_config)

    # 5. Example usage - Research and summarize a topic
    prompt_kwargs = {
        "topic": "artificial intelligence",
        "agents": {
            "research_agent": {
                "task": "Research the latest developments in artificial intelligence"
            },
            "summary_agent": {
                "task": "Create a concise summary of the research findings"
            }
        }
    }

    # Synchronous execution
    result = multirunner.call(
        prompt_kwargs=prompt_kwargs,
        model_kwargs={"temperature": 0.7},
        use_cache=False
    )
    print(f"Final result: {result}")

    # Asynchronous execution
    async_result = await multirunner.acall(
        prompt_kwargs=prompt_kwargs,
        model_kwargs={"temperature": 0.7},
        use_cache=False
    )
    print(f"Async result: {async_result}")

if __name__ == "__main__":
    main()
