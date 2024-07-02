"""Class prompt builder for LightRAG system prompt."""

from typing import Dict, Any, Optional, List, TypeVar
import logging
from functools import lru_cache

from jinja2 import Template, Environment, StrictUndefined, meta


from lightrag.core.component import Component
from lightrag.core.default_prompt_template import DEFAULT_LIGHTRAG_SYSTEM_PROMPT


logger = logging.getLogger(__name__)

T = TypeVar("T")


@lru_cache(None)
def get_jinja2_environment():
    r"""Helper function for Prompt component to get the Jinja2 environment with the default settings."""
    try:
        default_environment = Environment(
            undefined=StrictUndefined,
            trim_blocks=True,
            keep_trailing_newline=True,
            lstrip_blocks=True,
        )

        return default_environment
    except Exception as e:
        raise ValueError(f"Invalid Jinja2 environment: {e}")


class Prompt(Component):
    __doc__ = r"""Renders a text string(prompt) from a Jinja2 template string.

    In default, we use the :ref:`DEFAULT_LIGHTRAG_SYSTEM_PROMPT<core-default_prompt_template>`  as the template.

    Args:
        template (str, optional): The Jinja2 template string. Defaults to DEFAULT_LIGHTRAG_SYSTEM_PROMPT.
        preset_prompt_kwargs (Optional[Dict], optional): The preset prompt kwargs to fill in the variables in the prompt. Defaults to {}.

    Examples:
        >>> from core.prompt_builder import Prompt
        >>> prompt = Prompt(preset_prompt_kwargs={"task_desc_str": "You are a helpful assistant."})
        >>> print(prompt)
        >>> prompt.print_prompt_template()
        >>> prompt.print_prompt(context_str="This is a context string.")
        >>> prompt.call(context_str="This is a context string.")

        When examples_str itself is another template with variables, You can use another Prompt to render it.

        >>> EXAMPLES_TEMPLATE = r'''
        >>> {% if examples %}
        >>> {% for example in examples %}
        >>> {{loop.index}}. {{example}}
        >>> {% endfor %}
        >>> {% endif %}
        >>> '''
        >>> examples_prompt = Prompt(template=EXAMPLES_TEMPLATE)
        >>> examples_str = examples_prompt.call(examples=["Example 1", "Example 2"])
        >>> # pass it to the main prompt
        >>> prompt.print_prompt(examples_str=examples_str)
    """

    def __init__(
        self,
        template: Optional[str] = None,
        prompt_kwargs: Optional[Dict] = {},
    ):
        super().__init__()

        self.template = template or DEFAULT_LIGHTRAG_SYSTEM_PROMPT
        self.__create_jinja2_template()
        self.prompt_variables: List[str] = []
        for var in self._find_template_variables(self.template):
            self.prompt_variables.append(var)

        logger.info(f"{__class__.__name__} has variables: {self.prompt_variables}")

        self.prompt_kwargs = prompt_kwargs.copy()

    def __create_jinja2_template(self):
        r"""Create the Jinja2 template object."""
        try:
            self.jinja2_template: Template = get_jinja2_environment().from_string(
                self.template
            )
        except Exception as e:
            raise ValueError(f"Invalid Jinja2 template: {e}")

    def update_prompt_kwargs(self, **kwargs):
        r"""Update the initial prompt kwargs after Prompt is initialized."""
        self.prompt_kwargs.update(kwargs)

    def get_prompt_variables(self) -> List[str]:
        r"""Get the prompt kwargs."""
        return self.prompt_variables

    def is_key_in_template(self, key: str) -> bool:
        r"""Check if the key exists in the template."""
        return key in self.prompt_variables

    def _find_template_variables(self, template_str: str):
        """Automatically find all the variables in the template."""
        parsed_content = self.jinja2_template.environment.parse(template_str)
        return meta.find_undeclared_variables(parsed_content)

    def compose_prompt_kwargs(self, **kwargs) -> Dict:
        r"""Compose the final prompt kwargs by combining the initial and the provided kwargs at runtime."""
        composed_kwargs = {key: None for key in self.prompt_variables}
        if self.prompt_kwargs:
            composed_kwargs.update(self.prompt_kwargs)
        if kwargs:
            for key, _ in kwargs.items():
                if key not in composed_kwargs:
                    logger.warning(f"Key {key} does not exist in the prompt_kwargs.")
            composed_kwargs.update(kwargs)
        return composed_kwargs

    def print_prompt_template(self):
        r"""Print the template string."""
        print("Template:")
        print("-------")
        print(f"{self.template}")
        print("-------")

    def print_prompt(self, **kwargs):
        r"""Print the rendered prompt string using the preset_prompt_kwargs and the provided kwargs."""
        try:
            pass_kwargs = self.compose_prompt_kwargs(**kwargs)
            logger.debug(f"Prompt kwargs: {pass_kwargs}")

            prompt_str = self.jinja2_template.render(**pass_kwargs)
            print("Prompt:\n______________________")
            print(prompt_str)
        except Exception as e:
            raise ValueError(f"Error rendering Jinja2 template: {e}")

    def call(self, **kwargs) -> str:
        """
        Renders the prompt template with keyword arguments.
        """
        try:
            pass_kwargs = self.compose_prompt_kwargs(**kwargs)

            prompt_str = self.jinja2_template.render(**pass_kwargs)
            return prompt_str

        except Exception as e:
            raise ValueError(f"Error rendering Jinja2 template: {e}")

    def _extra_repr(self) -> str:
        s = f"template: {self.template}"
        if self.prompt_kwargs:
            s += f", prompt_kwargs: {self.prompt_kwargs}"
        if self.prompt_variables:
            s += f", prompt_variables: {self.prompt_variables}"
        return s

    @classmethod
    def from_dict(cls: type[T], data: Dict[str, Any]) -> T:
        obj = super().from_dict(data)
        # recreate the jinja2 template
        obj.jinja2_template = get_jinja2_environment().from_string(obj.template)
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """
        Get the dictionary representation of all the Prompt object's attributes, with sorting applied to
        dictionary keys and list elements to ensure consistent ordering.
        """
        exclude = ["jinja2_template"]  # unserializable object
        output = super().to_dict(exclude=exclude)
        return output
