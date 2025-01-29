import unittest
from unittest.mock import Mock


from adalflow.optim.parameter import Parameter
from adalflow.optim.gradient import GradientContext


class TestGradientContext(unittest.TestCase):
    def test_gradient_context_initialization(self):
        context = GradientContext(
            input_output="Sample context",
            response_desc="Sample response description",
            variable_desc="Sample variable description",
        )
        self.assertEqual(context.input_output, "Sample context")
        self.assertEqual(context.response_desc, "Sample response description")
        self.assertEqual(context.variable_desc, "Sample variable description")


#     def test_get_gradient_and_context_text(self):
#         expected_output = """
# Feedback 1.\n
# Here is a conversation:
# ____
# <CONVERSATION>Conversation context</CONVERSATION>
# This conversation is potentially part of a larger system. The output is used as <Response description>
# Here is the feedback we got for <Variable description> in the conversation:
#     <FEEDBACK>Gradient 2</FEEDBACK>
# """
#         self.assertEqual(
#             self.param1.get_gradient_and_context_text().strip(), expected_output.strip()
#         )


from adalflow.optim.text_grad.tgd_optimizer import TGDOptimizer


class TestUpdatePrompt(unittest.TestCase):
    def test_update_prompt(self):
        from adalflow.core.model_client import ModelClient

        # Setup
        param = Parameter(role_desc="Role description")
        # param.get_short_value = Mock(return_value="short value")
        param.get_gradient_and_context_text = Mock(
            return_value="gradient and context text"
        )

        # Create an instance of YourClass
        tgd = TGDOptimizer(model_client=ModelClient(), params=[param])
        tgd.in_context_examples = ["Example 1", "Example 2"]
        # tgd.get_g = Mock(return_value="Gradient memory text")
        tgd.do_in_context_examples = True
        tgd.do_constrained = True
        tgd.constraints = ["Some constraint text"]

        # Call the method
        user_prompt_kwargs = tgd._get_user_prompt_kwargs(param)
        result = tgd.llm_optimizer.get_prompt(**user_prompt_kwargs)

        # Check if each variable value is in the generated output
        # self.assertIn("Role description", result)
        # self.assertIn("short value", result)
        # self.assertIn("<start>", result)
        # self.assertIn("<end>", result)
        self.assertIn("Some constraint text", result)
        self.assertIn("Example 1", result)
        self.assertIn("Example 2", result)
        # self.assertIn("Gradient memory text", result)


if __name__ == "__main__":
    unittest.main()
