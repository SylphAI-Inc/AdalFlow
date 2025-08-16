# tests/test_agent_via_agent.py

import unittest
import types
from unittest.mock import patch

import adalflow.components.agent as agent_module
from adalflow.core.model_client import ModelClient


# --- Dummy stubs for dependencies ---


class FallbackModelClient(ModelClient):
    def call(self, api_kwargs, model_type):
        return None


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
        cache_path=None,
        use_cache=True,
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
        self.cache_path = cache_path
        self.use_cache = use_cache

    def get_prompt(self, **kwargs):
        return f"dummy-prompt({kwargs})"

    def __call__(self, *args, **kwargs):
        return types.SimpleNamespace(data="dummy response")


class TestAgent(unittest.TestCase):
    def setUp(self):
        # Stub out the real Generator where it's imported in the agent module
        self.gen_patcher = patch(
            "adalflow.components.agent.agent.Generator", DummyGenerator
        )
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
            model_kwargs={"temperature": 0.0, "model": "gpt-4o-mini"},
        )

        # Name & planner
        self.assertEqual(agent.name, "agent-with-fallback")
        self.assertIsInstance(agent.planner, DummyGenerator)

        # Tools include fallback when enabled
        tool_map = agent.tool_manager._context_map
        self.assertIsInstance(tool_map, dict)
        self.assertIn("llm_tool", tool_map)

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

        # No tools when llm_fallback is disabled and no custom tools provided
        tool_map = agent.tool_manager._context_map
        self.assertNotIn("llm_tool", tool_map)
        self.assertEqual(len(tool_map), 0)

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

    def test_agent_answer_data_type_attribute(self):
        """Test that Agent properly stores answer_data_type for Runner integration."""
        mc = FallbackModelClient()

        # Test with string type
        agent_str = agent_module.Agent(
            name="agent-str",
            tools=[],
            add_llm_as_fallback=False,
            model_client=mc,
            answer_data_type=str,
        )
        self.assertEqual(agent_str.answer_data_type, str)

        # Test with dict type
        agent_dict = agent_module.Agent(
            name="agent-dict",
            tools=[],
            add_llm_as_fallback=False,
            model_client=mc,
            answer_data_type=dict,
        )
        self.assertEqual(agent_dict.answer_data_type, dict)

    def test_agent_max_steps_attribute(self):
        """Test that Agent properly stores max_steps for Runner integration."""
        mc = FallbackModelClient()

        # Test default max_steps
        agent_default = agent_module.Agent(
            name="agent-default",
            tools=[],
            add_llm_as_fallback=False,
            model_client=mc,
        )
        self.assertEqual(agent_default.max_steps, 10)  # Default value

        # Test custom max_steps
        agent_custom = agent_module.Agent(
            name="agent-custom",
            tools=[],
            add_llm_as_fallback=False,
            model_client=mc,
            max_steps=5,
        )
        self.assertEqual(agent_custom.max_steps, 5)

    def test_agent_integration_with_runner_requirements(self):
        """Test that Agent has all required attributes for Runner integration."""
        mc = FallbackModelClient()
        agent = agent_module.Agent(
            name="agent-runner-ready",
            tools=[],
            add_llm_as_fallback=False,
            model_client=mc,
            answer_data_type=str,
            max_steps=8,
        )

        # Verify all required attributes for Runner are present
        self.assertTrue(hasattr(agent, "planner"))
        self.assertTrue(hasattr(agent, "tool_manager"))
        self.assertTrue(hasattr(agent, "max_steps"))
        self.assertTrue(hasattr(agent, "answer_data_type"))

        # Verify planner has required methods for Runner
        # The DummyGenerator uses __call__ method (like real Generator)
        self.assertTrue(hasattr(agent.planner, "__call__"))
        self.assertTrue(callable(agent.planner))

        # Verify tool_manager has required functionality for Runner
        self.assertTrue(hasattr(agent.tool_manager, "__call__"))
        self.assertTrue(callable(agent.tool_manager))

    def test_agent_training_mode_propagation(self):
        """Test that Agent's is_training method properly reflects planner training state."""
        mc = FallbackModelClient()
        agent = agent_module.Agent(
            name="agent-training",
            tools=[],
            add_llm_as_fallback=False,
            model_client=mc,
        )

        # Initially should not be in training mode
        self.assertFalse(agent.is_training())

        # Set planner to training mode
        agent.planner.training = True
        self.assertTrue(agent.is_training())

        # Disable training mode
        agent.planner.training = False
        self.assertFalse(agent.is_training())

    def test_flip_thinking_model_toggles_state(self):
        """Test that flip_thinking_model toggles the is_thinking_model state."""
        mc = FallbackModelClient()
        agent = agent_module.Agent(
            name="agent-thinking",
            tools=[],
            add_llm_as_fallback=False,
            model_client=mc,
            is_thinking_model=False,
        )

        # Initially should not be in thinking model mode
        self.assertFalse(agent.is_thinking_model)

        # Flip to thinking model
        agent.flip_thinking_model()
        self.assertTrue(agent.is_thinking_model)

        # Flip back to non-thinking model
        agent.flip_thinking_model()
        self.assertFalse(agent.is_thinking_model)

    def test_flip_thinking_model_recreates_planner(self):
        """Test that flip_thinking_model recreates the planner."""
        mc = FallbackModelClient()
        agent = agent_module.Agent(
            name="agent-planner-recreation",
            tools=[],
            add_llm_as_fallback=False,
            model_client=mc,
            is_thinking_model=False,
        )

        # Store reference to original planner
        original_planner = agent.planner
        original_planner_id = id(original_planner)

        # Flip thinking model state
        agent.flip_thinking_model()

        # Verify planner was recreated (different object)
        new_planner = agent.planner
        new_planner_id = id(new_planner)
        self.assertNotEqual(original_planner_id, new_planner_id)
        self.assertIsInstance(new_planner, DummyGenerator)

    def test_flip_thinking_model_preserves_other_attributes(self):
        """Test that flip_thinking_model preserves other agent attributes."""
        mc = FallbackModelClient()
        custom_kwargs = {"temperature": 0.5, "model": "test-model"}
        agent = agent_module.Agent(
            name="agent-preserve-attrs",
            tools=[],
            add_llm_as_fallback=False,
            model_client=mc,
            model_kwargs=custom_kwargs,
            max_steps=15,
            is_thinking_model=False,
        )

        # Store original values
        original_name = agent.name
        original_max_steps = agent.max_steps
        original_model_kwargs = agent.model_kwargs
        original_model_client = agent.model_client

        # Flip thinking model
        agent.flip_thinking_model()

        # Verify other attributes are preserved
        self.assertEqual(agent.name, original_name)
        self.assertEqual(agent.max_steps, original_max_steps)
        self.assertEqual(agent.model_kwargs, original_model_kwargs)
        self.assertEqual(agent.model_client, original_model_client)
        self.assertTrue(agent.is_thinking_model)  # This should be flipped

    def test_flip_thinking_model_with_initial_true(self):
        """Test flip_thinking_model when initially set to True."""
        mc = FallbackModelClient()
        agent = agent_module.Agent(
            name="agent-initial-thinking",
            tools=[],
            add_llm_as_fallback=False,
            model_client=mc,
            is_thinking_model=True,
        )

        # Initially should be in thinking model mode
        self.assertTrue(agent.is_thinking_model)

        # Flip to non-thinking model
        agent.flip_thinking_model()
        self.assertFalse(agent.is_thinking_model)

        # Flip back to thinking model
        agent.flip_thinking_model()
        self.assertTrue(agent.is_thinking_model)


if __name__ == "__main__":
    unittest.main()
