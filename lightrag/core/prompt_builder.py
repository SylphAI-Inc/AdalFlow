import jinja2.meta
from jinja2 import Template, Environment
import jinja2
from typing import Dict, Any, Optional, List
import logging
from functools import lru_cache

from lightrag.core.component import Component
from lightrag.core.default_prompt_template import DEFAULT_LIGHTRAG_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@lru_cache(None)
def get_jinja2_environment():
    r"""Helper function for Prompt component to get the Jinja2 environment with the default settings."""
    try:
        default_environment = Environment(
            undefined=jinja2.StrictUndefined,
            trim_blocks=True,
            keep_trailing_newline=True,
            lstrip_blocks=True,
        )

        return default_environment
    except Exception as e:
        raise ValueError(f"Invalid Jinja2 environment: {e}")


class Prompt(Component):
    __doc__ = r"""A component that renders a text string from a template using Jinja2 templates.

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
        *,
        template: str = DEFAULT_LIGHTRAG_SYSTEM_PROMPT,
        preset_prompt_kwargs: Optional[Dict] = {},  # preload the parameters
    ):
        super().__init__()

        self._template_string = template
        self.template: Template = None
        try:
            env = get_jinja2_environment()
            self.template = env.from_string(template)
        except Exception as e:
            raise ValueError(f"Invalid Jinja2 template: {e}")

        self.prompt_variables: List[str] = []
        for var in self._find_template_variables(self._template_string):
            self.prompt_variables.append(var)

        logger.info(f"{__class__.__name__} has variables: {self.prompt_variables}")

        self.preset_prompt_kwargs = preset_prompt_kwargs

    def update_preset_prompt_kwargs(self, **kwargs):
        r"""Update the preset prompt kwargs after Prompt is initialized."""
        self.preset_prompt_kwargs.update(kwargs)

    def get_prompt_variables(self) -> List[str]:
        r"""Get the prompt kwargs."""
        return self.prompt_variables

    def is_key_in_template(self, key: str) -> bool:
        r"""Check if the key exists in the template."""
        return key in self.prompt_variables

    def _find_template_variables(self, template_str: str):
        """Automatically find all the variables in the template."""
        parsed_content = self.template.environment.parse(template_str)
        return jinja2.meta.find_undeclared_variables(parsed_content)

    def compose_prompt_kwargs(self, **kwargs) -> Dict:
        r"""Compose the final prompt kwargs by combining the preset_prompt_kwargs and the provided kwargs."""
        composed_kwargs = {key: None for key in self.prompt_variables}
        if self.preset_prompt_kwargs:
            composed_kwargs.update(self.preset_prompt_kwargs)
        if kwargs:
            for key, _ in kwargs.items():
                if key not in composed_kwargs:
                    logger.warning(f"Key {key} does not exist in the prompt_kwargs.")
            composed_kwargs.update(kwargs)
        return composed_kwargs

    def print_prompt_template(self):
        r"""Print the template string."""
        print("Template:")
        print(f"-------")
        print(f"{self._template_string}")
        print(f"-------")

    def print_prompt(self, **kwargs):
        r"""Print the rendered prompt string using the preset_prompt_kwargs and the provided kwargs."""
        try:
            pass_kwargs = self.compose_prompt_kwargs(**kwargs)
            logger.debug(f"Prompt kwargs: {pass_kwargs}")

            prompt_str = self.template.render(**pass_kwargs)
            print("Prompt:")
            print(prompt_str)
        except Exception as e:
            raise ValueError(f"Error rendering Jinja2 template: {e}")

    def call(self, **kwargs) -> str:
        """
        Renders the prompt template with the provided variables.
        """
        try:
            pass_kwargs = self.compose_prompt_kwargs(**kwargs)

            prompt_str = self.template.render(**pass_kwargs)
            return prompt_str

        except Exception as e:
            raise ValueError(f"Error rendering Jinja2 template: {e}")

    def _extra_repr(self) -> str:
        s = f"template: {self._template_string}"
        if self.preset_prompt_kwargs:
            s += f", preset_prompt_kwargs: {self.preset_prompt_kwargs}"
        if self.prompt_variables:
            s += f", prompt_variables: {self.prompt_variables}"
        return s

    def to_dict(self) -> Dict[str, Any]:
        """
        Get the dictionary representation of all the Prompt object's attributes, with sorting applied to
        dictionary keys and list elements to ensure consistent ordering.
        """
        exclude = ["template"]
        output = super().to_dict(exclude=exclude)
        return output


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    prompt = Prompt(
        preset_prompt_kwargs={"task_desc_str": "You are a helpful assistant."}
    )
    print(prompt)
    prompt.print_prompt_template()
    prompt.print_prompt(context_str="This is a context string.")
    prompt.call(context_str="This is a context string.")
    states = prompt.state_dict()
    print(f"states: {states}")
    named_params = prompt.named_parameters()
    print(f"named_params: {named_params}")
    for name, param in named_params:
        print(f"{name}: {param}")

    # get dict of prompt
    prompt_dict = prompt.to_dict()
    print(f"prompt_dict: {prompt_dict}")
    prompt_state = prompt.state_dict()
    print(f"prompt_state: {prompt_state}")

    # EXAMPLES_TEMPLATE = r"""
    # {% if examples %}
    # {% for example in examples %}
    # {{loop.index}}. {{example}}
    # {% endfor %}
    # {% endif %}
    # """
    # examples_prompt = Prompt(template=EXAMPLES_TEMPLATE)
    # examples_str = examples_prompt.call(examples=["Example 1", "Example 2"])
    # prompt.print_prompt(examples_str=examples_str)
