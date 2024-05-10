from typing import Any, Dict, List, Optional
from core.data_classes import ModelType
from core.component import Component
from core.prompt_builder import Prompt
from core.functional import compose_model_kwargs
from core.api_client import APIClient

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
<EXAMPLES>
{{example_str}}
</EXAMPLES>
{% endif %}<</SYS>>
---------------------
{# chat history #}
{% if chat_history_str %}
Chat history:
{{chat_history_str}}
{% endif %}
User query: {{query_str}}
{#contex#}
{% if context_str %}
Context: {{context_str}}
{% endif %}
{# steps #}
{% if steps_str %}
{{steps_str}}
{% endif %}
{# assistant response #}
You:
"""
# TODO: special delimiters for different sections


class Generator(Component):
    model_type: ModelType = ModelType.LLM
    model_client: APIClient
    provider: str  # provider is model_provider
    prompt: Prompt
    output_processors: Optional[Component]

    def __init__(
        self,
        *,
        provider: Optional[
            str
        ] = None,  # TODO: not necessary as model_client has the provider already
        model_client: APIClient,
        model_kwargs: Optional[Dict] = {},
        prompt: Prompt = Prompt(DEFAULT_LIGHTRAG_PROMPT),
        preset_prompt_kwargs: Optional[Dict] = None,
        output_processors: Optional[Component] = None,
    ) -> None:
        super().__init__()
        self.model_kwargs = model_kwargs
        if "model" not in model_kwargs:
            raise ValueError(
                f"{type(self).__name__} requires a 'model' to be passed in the model_kwargs"
            )
        self.provider = provider
        self.prompt = prompt
        self.output_processors = output_processors
        self.model_client = model_client
        self.preset_prompt_kwargs = preset_prompt_kwargs
        # init the client
        self.model_client._init_sync_client()

    def train(self, *args, **kwargs):
        pass

    def _compose_lm_input_chat(self, **kwargs: Any) -> List[Dict]:
        """
        This combines the default lm input using Prompt, and the passed input. history, steps, etc.
        It builds the final chat input to the model.


        """
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

    def _compose_lm_input_non_chat(self, **kwargs: Any) -> str:
        """
        This combines the default lm input using Prompt, and the passed input. history, steps, etc.
        It builds the final chat input to the model.

        As
        """
        prompt_text = self.prompt.call(**kwargs)
        return prompt_text

    def compose_model_input(self, **kwargs) -> List[Dict]:

        return self._compose_lm_input_chat(**kwargs)

    def update_default_model_kwargs(self, **model_kwargs) -> Dict:
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
        print("Prompt:")
        print("-------")
        print(prompt)
        print("-------")
        return prompt

    # TODO: move this to output_processors
    def parse_completion(self, completion: Any) -> str:
        """
        Parse the completion to a structure your sytem standarizes. (here is str)
        # TODO: standardize the completion
        """
        return completion.choices[0].message.content

    def extra_repr(self) -> str:
        s = f"model_kwargs={self.model_kwargs} "
        if self.preset_prompt_kwargs:
            s += f", preset_prompt_kwargs={self.preset_prompt_kwargs} "
        return s

    def call(
        self,
        *,
        input: Any,
        prompt_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
    ) -> str:
        composed_model_kwargs = self.update_default_model_kwargs(**model_kwargs)
        if not self.model_client.sync_client:
            self.model_client.sync_client = self.model_client._init_sync_client()

        prompt_kwargs = self.compose_prompt_kwargs(**prompt_kwargs)
        # add the input to the prompt kwargs
        prompt_kwargs["query_str"] = input
        composed_messages = self.compose_model_input(**prompt_kwargs)
        print(f"composed_messages: {composed_messages}")
        print(f"composed_model_kwargs: {composed_model_kwargs}")
        # completion = self.sync_client.chat.completions.create(
        #     messages=composed_messages, **composed_model_kwargs
        # )
        # ToDO: the model client deal with backoff
        completion = self.model_client.call(
            input=composed_messages,
            model_kwargs=composed_model_kwargs,
            model_type=ModelType.LLM,
        )
        response = self.parse_completion(completion)
        if self.output_processors:
            response = self.output_processors.call(response)
        return response
