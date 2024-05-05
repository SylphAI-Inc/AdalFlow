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
        default_environment = Environment(undefined=jinja2.StrictUndefined)
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
        # first check if template is a valid Jinja2 template
        self._template_string = template
        self.template: Template = None
        try:
            env = get_jinja2_environment()
            self.template = env.from_string(template)
        except Exception as e:
            raise ValueError(f"Invalid Jinja2 template: {e}")

        self.variables: Dict[str, Any] = {}
        # store user-defined variables in the template
        # use _find_template_variables() to find all variables in the template and set the variables as input types
        for var in self._find_template_variables():
            self.variables[var] = None

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
            return self.template.render(**kwargs)
        except Exception as e:
            raise ValueError(f"Error rendering Jinja2 template: {e}")
