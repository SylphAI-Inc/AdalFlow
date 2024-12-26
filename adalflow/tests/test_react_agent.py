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
    """Test Agent with normal functions"""

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
            add_llm_as_fallback=False,
            model_client=self.mock_model_client,
            model_kwargs={"model": "gpt-3.5-turbo"},
        )

    def test_react_agent_initialization(self):
        self.assertEqual(self.react_agent.max_steps, 5)
        self.assertTrue(not self.react_agent.add_llm_as_fallback)
        self.assertEqual(
            len(self.react_agent.tool_manager.tools), 4
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
        # Simulate multiple steps for a complex query, each planner will return a FunctionExpression
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
            GeneratorOutput(
                data=FunctionExpression.from_function(
                    thought="Finish the task directly.",
                    func=self.react_agent._finish,
                    answer=12,
                )
            ),
        ]

        result = self.react_agent.call("Add 2 and 3, then multiply by 4.")
        print(f"result: {result}")
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
        # no action

        # check error raised
        with self.assertRaises(ValueError):

            result = self.react_agent.call("Simulate an error.")
            print(f"result 2: {result}")
            self.assertIn("Error occurred", result.answer)


from adalflow.optim.grad_component import GradComponent


class GradAdd(GradComponent):
    def __init__(self):
        super().__init__()

    def call(self, x, y):
        return x + y

    def forward(self, x, y):
        return f"{x + y} + forward"


class GradSub(GradComponent):
    def __init__(self):
        super().__init__()

    def call(self, x, y):
        return x - y

    def forward(self, x, y):
        return f"{x - y} + forward"


class TestReactAgentWithComponentASTool(unittest.TestCase):
    @patch("adalflow.components.model_client.openai_client.OpenAIClient", autospec=True)
    def setUp(self, MockOpenAIClient):
        """Set up the ReActAgent with GradComponents as tools."""
        self.add_component = GradAdd()
        self.sub_component = GradSub()

        self.tools = [
            FunctionTool(fn=self.add_component.__call__, component=self.add_component),
            FunctionTool(fn=self.sub_component.__call__, component=self.sub_component),
        ]

        self.mock_model_client = MockOpenAIClient.return_value
        self.agent = ReActAgent(
            tools=self.tools,
            max_steps=5,
            add_llm_as_fallback=False,
            model_client=self.mock_model_client,
            model_kwargs={"model": "gpt-3.5-turbo"},
        )

    def test_agent_with_eval_mode(self):
        """Test the agent's behavior when GradComponents are in eval mode."""
        # Ensure components start in eval mode
        self.assertFalse(self.add_component.training)
        self.assertFalse(self.sub_component.training)

        # Use agent to call addition tool
        result = self.agent.tool_manager.tools[0](3, 2)  # GradAdd in eval mode
        self.assertEqual(result.output, 5)

        # Use agent to call subtraction tool
        result = self.agent.tool_manager.tools[1](5, 3)  # GradSub in eval mode
        self.assertEqual(result.output, 2)

    def test_agent_with_train_mode(self):
        """Test the agent's behavior when GradComponents are in train mode."""
        # Set the agent to train mode, which should propagate to components
        self.agent.train()

        self.assertTrue(self.add_component.training)
        self.assertTrue(self.sub_component.training)
        # as the component is not directly registered in the agent, but passed to the tool manager, it will not be in training mode

        # Use agent to call addition tool in train mode
        result = self.agent.tool_manager.tools[0](3, 2)  # GradAdd in train mode
        self.assertEqual(result.output, "5 + forward")

        # Use agent to call subtraction tool in train mode
        result = self.agent.tool_manager.tools[1](5, 3)  # GradSub in train mode
        self.assertEqual(result.output, "2 + forward")

    def test_agent_switch_modes(self):
        """Test the agent's ability to switch between eval and train modes."""
        # Start in eval mode
        self.assertFalse(self.add_component.training)
        self.assertFalse(self.sub_component.training)

        # Switch to train mode
        self.agent.train()
        named_components = self.agent.named_components()
        for name, component in named_components:
            print(f"{name}: {component}")
        print(f"add_component: {self.add_component}")
        self.assertTrue(self.agent.tool_manager.training)

        # add component will have eval mode
        self.assertTrue(self.add_component.training)

        # the tools from the tool manager will be in training mode
        self.assertTrue(self.agent.tool_manager.tools[0].training)
        self.assertTrue(self.agent.tool_manager.tools[1].training)

        # back to eval mode
        self.agent.eval()
        self.assertFalse(self.add_component.training)
        self.assertFalse(self.sub_component.training)

        # tools from the tool manager will be in eval mode
        self.assertFalse(self.agent.tool_manager.tools[0].training)
        self.assertFalse(self.agent.tool_manager.tools[1].training)


if __name__ == "__main__":
    unittest.main()
