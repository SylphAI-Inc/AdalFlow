"""
AdalFlow RAG Agent Runner using AdalFlow's Runner and Agent components

This implementation demonstrates how to create a RAG (Retrieval-Augmented Generation) 
system using AdalFlow's core Agent and Runner components, with a DspyRetriever 
for document retrieval functionality.
"""

from typing import List, Optional
import dspy

print(dspy.__version__)

import adalflow as adal
from adalflow.components.agent import Agent, Runner
from adalflow.core.func_tool import FunctionTool
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.utils import setup_env

# Importing DspyRetriever directly to avoid config dependencies
from adalflow.core.retriever import Retriever
from adalflow.core.types import RetrieverOutput, ToolOutput


# DspyRetriever class definition (copied to avoid config dependencies)
class DspyRetriever(Retriever):
    def __init__(self, top_k: int = 3):
        super().__init__()
        self.top_k = top_k
        self.dspy_retriever = dspy.Retrieve(k=top_k)

    def call(
        self, input: str, top_k: Optional[int] = None, id: str = None
    ) -> ToolOutput:
        """Retrieves the top 2 passages using input as the query.
        Ensure you get all the context to answer the original question.
        r
        Args:
            input: The search query string
            id: Optional identifier for tracking

        Returns:
            List of retrieved document passages
        """

        k = top_k or self.top_k

        if not input:
            raise ValueError(f"Input cannot be empty, top_k: {k}")

        output = self.dspy_retriever(query=input, k=k)
        documents = output.passages

        return RetrieverOutput(
            query=input,
            documents=documents,
            doc_indices=[],
        )


# Configure DSPy retriever
colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url="http://20.102.90.50:2017/wiki17_abstracts"
)
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)


def main():
    """Main execution function implementing the specified usage pattern."""
    # Setup environment variables
    setup_env()

    # Initialize the retriever
    dspy_retriever = DspyRetriever(top_k=2)

    # Create tools for the agent
    tools = [
        FunctionTool(dspy_retriever.call),
    ]

    # Create the agent
    agent = Agent(
        name="RAGAgent",
        tools=tools,
        model_client=OpenAIClient(),
        model_kwargs={
            "model": "gpt-4o",
            "temperature": 0.3,
        },
        answer_data_type=str,
        max_steps=5,
    )

    # Create the runner
    runner = Runner(agent=agent, max_steps=5)

    print(f"Agent: {agent}")
    print(f"Runner: {runner}")

    question = "How many storeys are in the castle that David Gregory inherited?"

    # Set to training mode
    agent.train()

    # Execute the query
    output = runner.call(
        prompt_kwargs={
            "input_str": question,
        },
    )

    print(f"Final Answer: {output.answer}")
    print(f"Steps taken: {len(output.step_history)}")

    # Print step history
    for i, step in enumerate(output.step_history):
        print(f"Step {i+1}:")
        print(f"  Function: {step.function}")
        print(f"  Observation: {step.observation}")


if __name__ == "__main__":
    main()
