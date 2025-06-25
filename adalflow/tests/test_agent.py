# tests/test_agent_via_agent.py

import unittest
import types
from unittest.mock import patch

import adalflow.core.agent as agent_module
from adalflow.core.types import FunctionOutput


# --- Dummy stubs for dependencies ---


class FallbackModelClient:
    pass


class DummyGenerator:
    def __init__(
        self,
        model_client=None,
        model_kwargs=None,
        model_type=None,
        template=None,
        prompt_kwargs=None,
        output_processors=None,
        name=None,
        **kwargs,
    ):
        self.training = False
        self.model_client = model_client
        self.model_kwargs = model_kwargs
        self.model_type = model_type
        self.template = template
        self.prompt_kwargs = prompt_kwargs
        self.output_processors = output_processors
        self.name = name

    def get_prompt(self, **kwargs):
        return f"dummy-prompt({kwargs})"

    def __call__(self, *args, **kwargs):
        return types.SimpleNamespace(data="dummy response")


class TestAgent(unittest.TestCase):
    def setUp(self):
        # Stub out the real Generator
        self.gen_patcher = patch.object(agent_module, "Generator", DummyGenerator)
        self.gen_patcher.start()

        # No-op the fun_to_grad_component decorator so `finish` tool registers cleanly
        import adalflow.optim.grad_component as grad_comp

        self.decorator_patcher = patch.object(
            grad_comp, "fun_to_grad_component", lambda *args, **kwargs: (lambda fn: fn)
        )
        self.decorator_patcher.start()

    def tearDown(self):
        self.gen_patcher.stop()
        self.decorator_patcher.stop()

    def test_default_agent_with_llm_fallback(self):
        mc = FallbackModelClient()
        agent = agent_module.Agent(
            name="agent-with-fallback",
            tools=[],
            context_variables={"foo": "bar"},
            add_llm_as_fallback=True,
            model_client=mc,
        )

        # Name & planner
        self.assertEqual(agent.name, "agent-with-fallback")
        self.assertIsInstance(agent.planner, DummyGenerator)

        # Tools include both fallback and finish
        tool_map = agent.tool_manager._context_map
        self.assertIsInstance(tool_map, dict)
        self.assertIn("llm_tool", tool_map)
        self.assertIn("finish", tool_map)

        # Prompt delegation
        prompt = agent.get_prompt(example=123)
        self.assertEqual(prompt, "dummy-prompt({'example': 123})")

        # Training flag mirrored
        agent.planner.training = True
        self.assertTrue(agent.is_training())

    def test_default_agent_without_llm_fallback(self):
        mc = FallbackModelClient()
        agent = agent_module.Agent(
            name="agent-no-fallback",
            tools=[],
            context_variables={},
            add_llm_as_fallback=False,
            model_client=mc,
        )

        # Only the finish tool
        tool_map = agent.tool_manager._context_map
        self.assertNotIn("llm_tool", tool_map)
        self.assertIn("finish", tool_map)

        # Planner still present
        self.assertIsInstance(agent.planner, DummyGenerator)
        self.assertTrue(agent.get_prompt(x=1).startswith("dummy-prompt"))

    def test_agent_accepts_custom_tools(self):
        # A user-defined tool should appear in the map
        def custom_tool(x: int) -> int:
            return x * 2

        mc = FallbackModelClient()
        agent = agent_module.Agent(
            name="agent-tools",
            tools=[custom_tool],
            context_variables={},
            add_llm_as_fallback=False,
            model_client=mc,
        )

        tool_map = agent.tool_manager._context_map
        self.assertIn("custom_tool", tool_map)
        self.assertTrue(callable(tool_map["custom_tool"]))

        FunctionOutput

        self.assertEqual(tool_map["custom_tool"].call(10).output, 20)

    def test_model_kwargs_and_template_propagate_to_planner(self):
        mc = FallbackModelClient()
        agent = agent_module.Agent(
            name="agent-params",
            tools=[],
            context_variables={},
            add_llm_as_fallback=False,
            model_client=mc,
            model_kwargs={"beta": 99},
            template="my-custom-tpl",
        )

        # Our DummyGenerator recorded those args
        self.assertEqual(agent.planner.model_kwargs, {"beta": 99})
        self.assertEqual(agent.planner.template, "my-custom-tpl")

    def test_override_tool_manager_and_planner(self):
        # Create fakes that track identity
        class FakeTM:
            def __init__(self):
                self._context_map = {"x": "y"}

        fake_tm = FakeTM()
        fake_pg = DummyGenerator()

        agent = agent_module.Agent(
            name="agent-override", tool_manager=fake_tm, planner=fake_pg
        )

        self.assertIs(agent.tool_manager, fake_tm)
        self.assertIs(agent.planner, fake_pg)
        # Directly accessing the context map from tool manager
        self.assertEqual(agent.tool_manager._context_map, fake_tm._context_map)

    def test_context_variables_wired_into_tool_manager(self):
        mc = FallbackModelClient()
        ctx = {"user": "alice"}
        agent = agent_module.Agent(
            name="agent-context",
            tools=[],
            context_variables=ctx,
            add_llm_as_fallback=False,
            model_client=mc,
        )

        # The default TM stores it in its private additional_context
        self.assertEqual(
            agent.tool_manager._additional_context["context_variables"], ctx
        )


if __name__ == "__main__":
    unittest.main()
