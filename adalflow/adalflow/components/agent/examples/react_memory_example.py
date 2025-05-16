"""Example demonstrating ReAct agent with vector memory support."""

from adalflow.components.agent.react_agent import ReActAgent
from adalflow.core.func_tool import FunctionTool
from adalflow.components.model_client import OpenAIClient
from adalflow.core.types import Function
import logging

# from adalflow.components.memory import Memory
from adalflow.components.memory.memory import Memory

from dotenv import load_dotenv

load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Initialize model client
    model_client = OpenAIClient()
    model_kwargs = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
    }
    memory = Memory()

    # Define some example tools
    def calculate(expression: str, **kwargs) -> str:
        """Calculate the result of a mathematical expression."""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error calculating: {str(e)}"

    def get_factorial(n: int, **kwargs) -> str:
        """Calculate factorial of a number."""
        try:
            result = 1
            for i in range(1, n + 1):
                result *= i
            return str(result)
        except Exception as e:
            return f"Error calculating factorial: {str(e)}"

    def finish(answer: str, **kwargs) -> str:
        """Finish the conversation with a final answer."""
        return answer

    def extract_result(history: str, **kwargs) -> str:
        """Get the previous context."""
        return memory.call()

    def square(n: int, **kwargs) -> str:
        """Square a number."""
        return str(n * n)

    # Create example functions for the agent
    examples = [
        Function(
            thought="I need to calculate a simple arithmetic expression.",
            name="calculate",
            kwargs={"expression": "2 + 2"},
        ),
        Function(
            thought="I need to calculate the factorial of a number 5.",
            name="get_factorial",
            kwargs={"n": 5},
        ),
        Function(
            thought="I need to context data of previous conversation.",
            name="extract_result",
            kwargs={"history": "history"},
        ),
        Function(
            thought="I need to square a number 3.",
            name="square",
            kwargs={"n": 3},
        ),
    ]
    # Create function tools
    calc_tool = FunctionTool(calculate)
    factorial_tool = FunctionTool(get_factorial)
    extract_result_tool = FunctionTool(extract_result)
    square_tool = FunctionTool(square)

    # Create ReAct agent with vector memory
    agent = ReActAgent(
        tools=[
            calc_tool,
            factorial_tool,
            extract_result_tool,
            square_tool,
        ],
        model_client=model_client,
        model_kwargs=model_kwargs,
        add_llm_as_fallback=True,
        max_steps=5,
        examples=examples,
        debug=True,  # Enable debug output
    )

    # Example 1: Simple calculation
    logger.info("Example 1: Simple calculation")
    print("MEMORY_CALL", memory.call())
    agent.context_variables = {"context_variables": {"history": memory.call()}}
    result1 = agent("What is 2 + 1?")
    logger.info(f"Result 1: {result1.answer}")
    memory.add_dialog_turn("What is 2 + 1?", result1.step_history)

    # Example 2: Using previous context
    print("MEMORY_CALL", memory.call())
    logger.info("\nExample 2: Using previous context")
    agent.context_variables = {"context_variables": {"history": memory.call()}}
    result2 = agent("what is the result of the previous question?")
    logger.info(f"Result 2: {result2.answer}")
    memory.add_dialog_turn(
        "what is the result of the previous question?", result2.step_history
    )

    # Example 2: Using previous context
    print("MEMORY_CALL", memory.call())
    logger.info("\nExample 3: Factorial of previous final result")
    agent.context_variables = {"context_variables": {"history": memory.call()}}
    result3 = agent("what is the square of the previous final result?")
    logger.info(f"Result 3: {result3.answer}")
    memory.add_dialog_turn(
        "what is the square of the previous final result?", result3.step_history
    )


if __name__ == "__main__":
    main()
