import unittest
from unittest.mock import Mock, patch
from adalflow.core.func_tool import FunctionTool
from adalflow.core.types import FunctionExpression, GeneratorOutput
from adalflow.components.agent.react import ReActAgent
from adalflow.components.model_client.openai_client import OpenAIClient


# Mock tools for testing
def mock_add(a: int, b: int) -> int:
    return a + b


def mock_multiply(a: int, b: int) -> int:
    return a * b


def mock_simple_tool(input: str) -> str:
    return f"Processed: {input}"


class TestReActAgent(unittest.TestCase):
    def setUp(self):
        # Mock OpenAIClient
        self.mock_model_client = Mock(spec=OpenAIClient)

        # Initialize ReActAgent with mocked tools and model client
        self.tools = [
            FunctionTool(mock_add),
            FunctionTool(mock_multiply),
            FunctionTool(mock_simple_tool),
        ]
        self.react_agent = ReActAgent(
            tools=self.tools,
            max_steps=5,
            add_llm_as_fallback=True,
            model_client=self.mock_model_client,
            model_kwargs={"model": "gpt-3.5-turbo"},
        )

    def test_react_agent_initialization(self):
        self.assertEqual(self.react_agent.max_steps, 5)
        self.assertTrue(self.react_agent.add_llm_as_fallback)
        self.assertEqual(
            len(self.react_agent.tool_manager.tools), 5
        )  # 3 tools + finish + fallback

    @patch.object(ReActAgent, "planner", create=True)
    def test_simple_query_execution(self, mock_planner):
        # Simulate a valid JSON-serializable response from the planner
        mock_planner.return_value = GeneratorOutput(
            data=FunctionExpression.from_function(
                thought="Finish the task directly.",
                func=self.react_agent._finish,
                answer="Simple answer",
            )
        )

        result = self.react_agent.call("What is 2 + 2?")
        self.assertEqual(result.answer, "Simple answer")

    @patch.object(ReActAgent, "planner", create=True)
    def test_complex_query_execution(self, mock_planner):
        # Simulate multiple steps for a complex query
        mock_planner.side_effect = [
            GeneratorOutput(
                data=FunctionExpression.from_function(
                    thought="Divide the task into subqueries.", func=mock_add, a=2, b=2
                )
            ),
            GeneratorOutput(
                data=FunctionExpression.from_function(
                    thought="Multiply the results.", func=mock_multiply, a=4, b=3
                )
            ),
        ]

        result = self.react_agent.call("Add 2 and 3, then return the result.")
        self.assertEqual(result.answer, 12)

    @patch.object(ReActAgent, "planner", create=True)
    def test_error_handling(self, mock_planner):
        # Simulate an error scenario
        mock_planner.return_value = GeneratorOutput(
            data={
                "thought": "Encountered an error.",
                "function": {"name": "finish", "args": {"answer": "Error occurred"}},
            }
        )

        result = self.react_agent.call("Simulate an error.")
        self.assertIn("Error occurred", result.answer)


if __name__ == "__main__":
    unittest.main()
