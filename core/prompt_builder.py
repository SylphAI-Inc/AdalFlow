import jinja2.meta
from jinja2 import Template, Environment
import jinja2
from typing import Dict, Any, Optional

from functools import lru_cache

from core.component import Component

from core.default_prompt_template import DEFAULT_LIGHTRAG_PROMPT


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

    def __init__(
        self,
        *,
        template: str = DEFAULT_LIGHTRAG_PROMPT,
        preset_prompt_kwargs: Optional[Dict] = {},
    ):
        super().__init__()
        self._template_string = template
        self.template: Template = None
        try:
            env = get_jinja2_environment()
            self.template = env.from_string(template)
        except Exception as e:
            raise ValueError(f"Invalid Jinja2 template: {e}")

        self.prompt_kwargs: Dict[str, Any] = {}
        for var in self._find_template_variables(self._template_string):
            self.prompt_kwargs[var] = None
        self.preset_prompt_kwargs = preset_prompt_kwargs
        # iterate each key value in preset and to discover potential variables
        for key, value in preset_prompt_kwargs.items():
            for var in self._find_template_variables(value):
                if var not in self.prompt_kwargs:
                    self.prompt_kwargs[var] = None

    def update_preset_prompt_kwargs(self, **kwargs):
        self.preset_prompt_kwargs.update(kwargs)

    def _find_template_variables(self, template_str: str):
        """Automatically find all the variables in the template."""
        parsed_content = self.template.environment.parse(template_str)
        return jinja2.meta.find_undeclared_variables(parsed_content)

    def compose_prompt_kwargs(self, **kwargs) -> Dict:
        composed_kwargs = self.prompt_kwargs.copy()
        if self.preset_prompt_kwargs:
            composed_kwargs.update(self.preset_prompt_kwargs)
        # runtime kwargs will overwrite the preset kwargs
        composed_kwargs.update(kwargs)
        return composed_kwargs

    def print_prompt(self, **kwargs):
        r"""To better visualize the prompt: as close as the final prompt string.
        For task-specific variables, such as task_desc_str, tools_str, we replace the them with the actual values from the preset_prompt_kwargs.
        For per-query variables such as query_str, chat_history_str, we leave it as it is in the template using the custom filter none_filter.
        """
        try:
            pass_kwargs = self.compose_prompt_kwargs(**kwargs)

            print(f"pass_kwargs: {pass_kwargs}  ")

            prompt_str = self.template.render(**pass_kwargs)
            print("Prompt:")
            print("-------")
            print(prompt_str)
            print("-------")
        except Exception as e:
            raise ValueError(f"Error rendering Jinja2 template: {e}")

    def print_prompt_template(self):
        print("Template:")
        print(f"-------")
        print(f"{self._template_string}")
        print(f"-------")

    def call(self, **kwargs) -> str:
        """
        Renders the prompt template with the provided variables.
        TODO: if there are submodules,
        """
        try:
            pass_kwargs = self.compose_prompt_kwargs(**kwargs)

            prompt_str = self.template.render(**pass_kwargs)
            return prompt_str

        except Exception as e:
            raise ValueError(f"Error rendering Jinja2 template: {e}")

    def extra_repr(self) -> str:
        s = f"template: {self._template_string}"
        if self.preset_prompt_kwargs:
            s += f", preset_prompt_kwargs: {self.preset_prompt_kwargs}"
        if self.prompt_kwargs:
            s += f", prompt_kwargs: {self.prompt_kwargs}"
        return s
