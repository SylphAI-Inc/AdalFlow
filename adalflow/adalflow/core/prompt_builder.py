"""Class prompt builder for AdalFlow system prompt."""

from typing import Dict, Any, Optional, List, TypeVar, Union
import logging
from functools import lru_cache

from jinja2 import Template, Environment, StrictUndefined, meta


from adalflow.core.default_prompt_template import DEFAULT_ADALFLOW_SYSTEM_PROMPT
from adalflow.optim.parameter import Parameter
from adalflow.core.component import DataComponent


logger = logging.getLogger(__name__)

T = TypeVar("T")


class Prompt(DataComponent):
    __doc__ = r"""Renders a text string(prompt) from a Jinja2 template string.

    In default, we use the :ref:`DEFAULT_ADALFLOW_SYSTEM_PROMPT<core-default_prompt_template>`  as the template.

    Args:
        template (str, optional): The Jinja2 template string. Defaults to DEFAULT_ADALFLOW_SYSTEM_PROMPT.
        preset_prompt_kwargs (Optional[Dict], optional): The preset prompt kwargs to fill in the variables in the prompt. Defaults to {}.

    Examples:
        >>> from core.prompt_builder import Prompt
        >>> prompt = Prompt(prompt_kwargs={"task_desc_str": "You are a helpful assistant."})
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
    # # save these two fields for serialization, using to_dict and from_dict
    # template: str = field(
    #     default=DEFAULT_ADALFLOW_SYSTEM_PROMPT,
    #     metadata={"desc": "The Jinja2 template string."},
    # )
    # prompt_kwargs: Dict[str, Parameter] = field(
    #     default_factory=dict, metadata={"desc": "The preset prompt kwargs."}
    # )
    # prompt_va

    def __init__(
        self,
        template: Optional[str] = None,
        prompt_kwargs: Optional[Dict[str, Union[Any, Parameter]]] = {},
    ):
        super().__init__()

        self.template = template or DEFAULT_ADALFLOW_SYSTEM_PROMPT
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
            # for key, _ in kwargs.items():
            #     if key not in composed_kwargs:
            #         logger.debug(f"Key {key} does not exist in the prompt_kwargs.")
            composed_kwargs.update(kwargs)
        return composed_kwargs

    def print_prompt_template(self):
        r"""Print the template string."""
        print("Template:")
        print("-------")
        print(f"{self.template}")
        print("-------")

    def print_prompt(self, **kwargs) -> str:
        r"""Print the rendered prompt string using the preset_prompt_kwargs and the provided kwargs."""
        try:
            pass_kwargs = self.compose_prompt_kwargs(**kwargs)
            pass_kwargs = self._convert_prompt_kwargs_to_str(pass_kwargs)
            logger.debug(f"Prompt kwargs: {pass_kwargs}")

            prompt_str = self.jinja2_template.render(**pass_kwargs)
            print("Prompt:\n______________________")
            print(prompt_str)
            return prompt_str
        except Exception as e:
            raise ValueError(f"Error rendering Jinja2 template: {e}")

    # def __call__(self, *args: Any, **kwds: Any) -> Any:
    #     return self.call(*args, **kwds)

    def _deep_render(self, value: Any, kwargs: Dict[str, Any]) -> Any:
        """Recursively render *value* with the same kwargs.
            • If value is another Prompt  → call it with kwargs
            • If value is a jinja2.Template
            • Otherwise                   → return as‑is

        Note:
        - Avoid passing Prompt/Template objects to the prompt kwargs to avoid circular references
        - Ensure we dont pass used a nested prompt/template in a sequence/dict/list
        """
        from jinja2 import Template

        # 1. filter out the prompt/template objects from kwargs to avoid circular references
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if not isinstance(v, (Prompt, Template))
        }

        if isinstance(value, Prompt):
            output =  value.call(**filtered_kwargs)
            return output

        if isinstance(value, Template):
            output = value.render(**filtered_kwargs)
            return output

        # if isinstance(value, str) and ("{{" in value or "{%" in value):
        #     # Treat raw strings as one‑off templates
        #     try:
        #         tpl = get_jinja2_environment().from_string(value)
        #         return tpl.render(**filtered_kwargs)
        #     except UndefinedError:
        #         # leave unresolved – outer template may supply the missing vars
        #         return value
        # if isinstance(value, list):
        #     return [self._deep_render(v, filtered_kwargs) for v in value]

        # if isinstance(value, dict):
        #     return {k: self._deep_render(v, filtered_kwargs) for k, v in value.items()}

        return value

    def call(self, **kwargs) -> str:
        """
        Renders the prompt template with keyword arguments. Allow None values.
        """
        try:
            pass_kwargs = self.compose_prompt_kwargs(**kwargs)
            # print(f"Prompt kwargs: {pass_kwargs}")
            pass_kwargs = self._convert_prompt_kwargs_to_str(pass_kwargs)
            # print(f"Prompt kwargs after conversion: {pass_kwargs}")
            prompt_str = self.jinja2_template.render(**pass_kwargs)
            return prompt_str

        except Exception as e:
            raise ValueError(f"Error rendering Jinja2 template: {e}")

    def _extra_repr(self) -> str:
        s = f"template: {self.template}"
        prompt_kwargs_str = self._convert_prompt_kwargs_to_str(self.prompt_kwargs)
        if prompt_kwargs_str:
            s += f", prompt_kwargs: {prompt_kwargs_str}"
        if self.prompt_variables:
            s += f", prompt_variables: {self.prompt_variables}"
        return s

    def __repr__(self) -> str:
        s = f"template: {self.template}"
        prompt_kwargs_str = self._convert_prompt_kwargs_to_str(self.prompt_kwargs)
        if prompt_kwargs_str:
            s += f", prompt_kwargs: {prompt_kwargs_str}"
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

    def _convert_prompt_kwargs_to_str(self, prompt_kwargs: Dict) -> Dict[str, str]:
        r"""Convert the prompt_kwargs to a dictionary with string values."""
        prompt_kwargs_str: Dict[str, str] = {}

        for key, p in prompt_kwargs.items():

            if isinstance(p, Parameter):

                prompt_kwargs_str[key] = p.data
            elif isinstance(p, list):
                prompt_kwargs_str[key] = [
                    (
                        p_elem.data_in_prompt(p_elem)
                        if isinstance(p_elem, Parameter)
                        else p_elem
                    )
                    for p_elem in p
                ]

            else:
                prompt_kwargs_str[key] = p

        # Pass‑2: recursively render any templates ── new behaviour
        # Create a copy to avoid circular reference during deep rendering
        context_for_render = prompt_kwargs_str.copy()
        for k, v in list(prompt_kwargs_str.items()):
            prompt_kwargs_str[k] = self._deep_render(v, context_for_render)

        return prompt_kwargs_str


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


if __name__ == "__main__":

    import adalflow as adal

    def test_template():
        template = r"""<START_OF_SYSTEM_MESSAGE>{{ task_desc_str }}<END_OF_SYSTEM_MESSAGE>
{# tools #}
{% if tools %}
<TOOLS>
{% for tool in tools %}
{{loop.index}}. {{ tool }}
{% endfor %}
</TOOLS>{% endif %}
<START_OF_USER>{{ input_str }} <END_OF_USER>"""

        task_desc_str = "You are a helpful assitant"

        tools = ["google", "wikipedia", "wikidata"]

        prompt = adal.Prompt(
            template=template,
            prompt_kwargs={
                "task_desc_str": task_desc_str,
                "tools": tools,
            },
        )

        print(prompt(input_str="What is the capital of France?"))

        to_dict = prompt.to_dict()

        prompt_restructured = adal.Prompt.from_dict(to_dict)

        print(to_dict)
        print(prompt_restructured)

    def test_nested_templates():
        OUTER = """
    <START_OF_SYSTEM_MESSAGE>{{ task_desc_str }}<END_OF_SYSTEM_MESSAGE>
    Will write files under {{ work_dir }}

    Examples
    ---------
    {{ examples_block }}

    <START_OF_USER>{{ input_str }}<END_OF_USER>
    """

        EXAMPLES_TEMPLATE = r"""
        {% for eg in examples %}
        {{ loop.index }}. {{ eg }}
        {% endfor %}
        """

        # create once and reuse
        inner_examples = Prompt(template=EXAMPLES_TEMPLATE)

        prompt = Prompt(template=OUTER)

        msg = prompt.call(
            # outer‑level vars
            task_desc_str="You are a helpful assistant.",
            work_dir="/tmp/run‑42",
            # nested Prompt object – no manual render needed
            examples_block=inner_examples,
            # variables that *only* the inner template needs
            examples=["Paris is the capital of France", "Rome is the capital of Italy"],
            # user input
            input_str="List those capitals again, please.",
        )

        print(msg)

    # test_template()
    test_nested_templates()
