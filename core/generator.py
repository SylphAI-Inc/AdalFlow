from typing import Any, Dict, List, Optional
from core.data_classes import ModelType
from core.component import Component
from core.prompt_builder import Prompt
from core.functional import compose_model_kwargs

DEFAULT_LIGHTRAG_PROMPT = r"""
<<SYS>>{# task desc #}
{% if task_desc_str %}
{{task_desc_str}}
{% endif %}
{# tools #}
{% if tools_str %}
{{tools_str}}
{% endif %}
{# example #}
{% if example_str %}
{{example_str}}
{% endif %}<</SYS>>
---------------------
User query: {{query_str}}
{#contex#}
{% if context_str %}
Context: {{context_str}}
{% endif %}
{# chat history #}
{% if chat_history_str %}
{{chat_history_str}}
{% endif %}
{# steps #}
{% if steps_str %}
{{steps_str}}
{% endif %}
{# assistant response #}
You:
"""


class Generator(Component):
    r"""
    A general class for all LLM models. Assume it meets the OpenAI API standard.
    """

    type: ModelType = ModelType.LLM

    provider: str
    prompt: Prompt
    output_processors: Optional[Component]

    def __init__(
        self,
        *,
        provider: Optional[str],
        prompt: Prompt,
        preset_prompt_kwargs: Optional[Dict] = None,
        output_processors: Optional[Component],
        model_kwargs: Optional[Dict] = {},
    ) -> None:
        super().__init__()
        self.model_kwargs = model_kwargs
        self.provider = provider
        self.prompt = prompt
        self.output_processors = output_processors
        self.preset_prompt_kwargs = preset_prompt_kwargs
        self.sync_client = self._init_sync_client()

    def _init_sync_client(self):
        r"""Initialize your client and potentially api keys here."""
        raise NotImplementedError(
            f"Model {type(self).__name__} is missing the required '_init_client' method."
        )

    def _init_async_client(self):
        pass

    def _componse_lm_input_chat(self, **kwargs: Any) -> List[Dict]:
        """
        This combines the default lm input using Prompt, and the passed input. history, steps, etc.
        It builds the final chat input to the model.


        """
        # ensure self.prompt is set
        if not hasattr(self, "prompt") or not self.prompt:
            raise ValueError(
                f"{type(self).__name__} requires a 'prompt' to be set before calling the model."
            )
        current_role: str = kwargs.get("role", "system")
        previous_messages: List[Dict] = kwargs.get("previous_messages", [])
        prompt_text = self.prompt.call(**kwargs)
        # llm input or the api's input
        messages = previous_messages + [{"role": current_role, "content": prompt_text}]
        return messages

    def _componse_lm_input_non_chat(self, **kwargs: Any) -> str:
        """
        This combines the default lm input using Prompt, and the passed input. history, steps, etc.
        It builds the final chat input to the model.

        As
        """
        prompt_text = self.prompt.call(**kwargs)
        return prompt_text

    def compose_model_input(self, **kwargs) -> List[Dict]:

        return self._componse_lm_input_chat(**kwargs)

    def compose_model_kwargs(self, **model_kwargs) -> Dict:
        r"""
        The model configuration exclude the input itself.
        Combine the default model, model_kwargs with the passed model_kwargs.
        Example:
        model_kwargs = {"temperature": 0.5, "model": "gpt-3.5-turbo"}
        self.model_kwargs = {"model": "gpt-3.5"}
        combine_kwargs(model_kwargs) => {"temperature": 0.5, "model": "gpt-3.5-turbo"}

        """
        return compose_model_kwargs(self.model_kwargs, model_kwargs)

    def compose_prompt_kwargs(self, **kwargs) -> Dict:
        composed_kwargs = (
            self.preset_prompt_kwargs.copy() if self.preset_prompt_kwargs else {}
        )
        composed_kwargs.update(kwargs)
        return composed_kwargs

    def print_prompt(self, **kwargs) -> str:
        composed_kwargs = self.compose_prompt_kwargs(**kwargs)
        prompt = self.prompt.call(**composed_kwargs)
        print(f"prompt: {prompt}")
        return prompt

    def parse_completion(self, completion: Any) -> str:
        """
        Parse the completion to a structure your sytem standarizes. (here is str)
        # TODO: standardize the completion
        """
        return completion.choices[0].message.content
