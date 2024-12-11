from adalflow.core import Component, Prompt

# Parameter

from unittest.mock import Mock
from unittest import TestCase
import pytest


class ComponentMissSuperInit(Component):
    def __init__(self, name, age):
        # super().__init__()
        self.name = name
        self.age = age
        self.height = 180
        # self.generator = Generator(
        #     model_client=GroqAPIClient(),
        #     model_kwargs={"model": "llama3-8b-8192"},
        # )
        self.mock_prompt = Mock(spec=Prompt)

    def call(self, query: str) -> str:
        return f"Hello {query}"


class ComponentWithSuperInit(Component):
    def __init__(self, name, age):
        super().__init__()
        self.name = name
        self.age = age
        # self.generator = Generator(
        #     model_client=GroqAPIClient(),
        #     model_kwargs={"model": "llama3-8b-8192"},
        # )
        # self.height = Parameter[int](data=180)
        # self.mock_prompt = Prompt(template="Hello {{input}}")
        # self.mock_prompt.register_parameter("input", Parameter[str](data="John"))

    def call(self, query: str) -> str:
        return f"Hello {query}"


class TestComponent(TestCase):
    def test_component_missing_super_init(self):
        with pytest.raises(AttributeError):
            a = ComponentMissSuperInit("John", 30)  # noqa: F841

    def test_component_with_super_init(self):
        a = ComponentWithSuperInit("John", 30)  # noqa: F841

        # 1. check named_parameters
        # named_parameters = dict(a.named_parameters())

        # expected_named_parameters = {
        #     "height": a.height,
        #     "mock_prompt.input": a.mock_prompt.input,
        # }
        # self.assertEqual(named_parameters, expected_named_parameters)

        # # 2. Check parameters
        # parameters = list(a.parameters())
        # expected_parameters = [a.height, a.mock_prompt.input]
        # self.assertEqual(parameters, expected_parameters)

        # # 3. check named_parameters with recursive = False
        # named_parameters = dict(a.named_parameters(recursive=False))
        # expected_named_parameters = {"height": a.height}
        # self.assertEqual(named_parameters, expected_named_parameters)

        # # 4. Check parameters with recursive = False
        # parameters = list(a.parameters(recursive=False))
        # expected_parameters = [a.height]
        # self.assertEqual(parameters, expected_parameters)

    def test_component_components(self):
        a = ComponentWithSuperInit("John", 30)
        # 1. register a subcomponent
        sub_component = Mock(spec=Component)
        sub_component.named_components.return_value = iter([])
        a.register_component("mock_prompt", sub_component)

        # 2. Ensure we can find the subcomponent in a._components
        self.assertIn("mock_prompt", a._components)

        # 3. Get the subcomponent with name "mock_prompt"
        mock_prompt = a.get_subcomponent("mock_prompt")
        self.assertEqual(mock_prompt, sub_component)

        # 4. Do named_components
        named_components = dict(a.named_components())
        expected_named_components = {"": a}
        self.assertEqual(named_components, expected_named_components)
