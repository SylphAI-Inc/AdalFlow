import jinja2.meta
from core.component import Component
from jinja2 import Template, Environment
import jinja2
from typing import Dict, Any

from functools import lru_cache


# cache the environment for faster template rendering
@lru_cache(None)
def get_jinja2_environment():
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
    """
    A component that renders a text string from a template using Jinja2 templates.
    As inherited from component, it is highly flexible, it can have
    other subcomponents which might do things like query expansion, document retrieval if you prefer
    to have it here.
    """

    def __init__(self, template: str):
        super().__init__()
        self._template_string = template
        self.template: Template = None
        try:
            env = get_jinja2_environment()
            self.template = env.from_string(template)
        except Exception as e:
            raise ValueError(f"Invalid Jinja2 template: {e}")

        self.prompt_kwargs: Dict[str, Any] = {}
        for var in self._find_template_variables():
            self.prompt_kwargs[var] = None

    def _find_template_variables(self):
        """Automatically find all the variables in the template."""
        parsed_content = self.template.environment.parse(self._template_string)
        return jinja2.meta.find_undeclared_variables(parsed_content)

    def call(self, **kwargs) -> str:
        """
        Renders the prompt template with the provided variables.
        TODO: if there are submodules,
        """
        try:
            pass_kwargs = self.prompt_kwargs.copy()
            pass_kwargs.update(kwargs)

            prompt_str = self.template.render(**pass_kwargs)
            return prompt_str

        except Exception as e:
            raise ValueError(f"Error rendering Jinja2 template: {e}")

    def extra_repr(self) -> str:
        return f"template: {self._template_string}"
