from typing import Any, Dict, List, Optional, Tuple
from core.data_classes import ModelType
from core.component import Component
from core.prompt_builder import Prompt
from core.functional import compose_model_kwargs
from core.api_client import APIClient
from core.default_prompt_template import DEFAULT_LIGHTRAG_SYSTEM_PROMPT


GeneratorInputType = str
GeneratorOutputType = Any


class Generator(Component):
    """
    An orchestrator component that combines the Prompt and the API client to generate text from a prompt.
    Additionally, it allows you to pass the output_processors to further parse the output from the model. Thus the arguments are almost a combination of that of Prompt and APIClient.

    It takes the user query as input in string format, and returns the response or processed response.
    """

    model_type: ModelType = ModelType.LLM
    model_client: APIClient  # for better type checking

    def __init__(
        self,
        *,
        model_client: APIClient,
        model_kwargs: Dict[str, Any] = {},
        # args for the prompt
        template: str = DEFAULT_LIGHTRAG_SYSTEM_PROMPT,
        preset_prompt_kwargs: Optional[Dict] = None,  # manage the prompt kwargs
        output_processors: Optional[Component] = None,
    ) -> None:
        r"""The default prompt is set to the DEFAULT_LIGHTRAG_SYSTEM_PROMPT. It has the following variables:
        - task_desc_str
        - tools_str
        - example_str
        - chat_history_str
        - query_str
        - context_str
        - steps_str
        You can preset the prompt kwargs to fill in the variables in the prompt using preset_prompt_kwargs.
        But you can replace the prompt and set any variables you want and use the preset_prompt_kwargs to fill in the variables.
        """
        super().__init__()
        self.model_kwargs = model_kwargs
        if "model" not in model_kwargs:
            raise ValueError(
                f"{type(self).__name__} requires a 'model' to be passed in the model_kwargs"
            )
        # init the model client
        self.model_client = model_client()
        self.system_prompt = Prompt(
            template=template, preset_prompt_kwargs=preset_prompt_kwargs
        )

        self.output_processors = output_processors

    def train(self, *args, **kwargs):
        pass

    def _compose_lm_input_chat(self, input: str, **kwargs: Any) -> List[Dict]:
        """
        Forms the final messages to LLM chat model.

        example:
        {
            "role": "system",

        }
        """
        if not hasattr(self, "system_prompt") or not self.system_prompt:
            raise ValueError(
                f"{type(self).__name__} requires a 'system_prompt' to be set before calling the model."
            )

        system_prompt_text = self.system_prompt.call(**kwargs).strip()
        messages: List[Dict[str, str]] = []
        if system_prompt_text and system_prompt_text != "":
            messages = [{"role": "system", "content": system_prompt_text}]
        user_message = {"role": "user", "content": input}
        messages.append(user_message)

        return messages

    def _compose_lm_input_non_chat(self, **kwargs: Any) -> str:
        """
        This combines the default lm input using Prompt, and the passed input. history, steps, etc.
        It builds the final chat input to the model.

        As
        """
        prompt_text = self.system_prompt.call(**kwargs)
        return prompt_text

    # TODO: not used for now
    def compose_model_input(self, input: str, **kwargs) -> List[Dict]:

        return self._compose_lm_input_chat(input=input, **kwargs)

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

    def print_prompt(self, **kwargs) -> str:
        self.system_prompt.print_prompt(**kwargs)

    # TODO: move this potntially to api_client
    def parse_completion(self, completion: Any) -> str:
        """
        Parse the completion to a structure your sytem standarizes. (here is str)
        # TODO: standardize the completion
        """
        return completion.choices[0].message.content

    def extra_repr(self) -> str:
        s = f"model_kwargs={self.model_kwargs}, model_type={self.model_type}"
        return s

    def _pre_call(
        self,
        input: str,
        prompt_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
    ) -> Tuple[List[Dict], Dict]:
        r"""Compose the input and model_kwargs before calling the model."""
        composed_model_kwargs = self.update_default_model_kwargs(**model_kwargs)
        composed_messages = self.compose_model_input(input=input, **prompt_kwargs)
        print(f"composed_messages: {composed_messages}")
        # TODO: change everything to logger(info)
        return composed_messages, composed_model_kwargs

    def _post_call(self, completion: Any) -> GeneratorOutputType:
        r"""Parse the completion and process the output."""
        response = self.parse_completion(completion)
        if self.output_processors:
            response = self.output_processors(response)
        return response

    def call(
        self,
        input: str,
        prompt_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
    ) -> GeneratorOutputType:
        r"""Call the model with the input(user_query) and model_kwargs."""
        composed_messages, composed_model_kwargs = self._pre_call(
            input, prompt_kwargs, model_kwargs
        )
        completion = self.model_client.call(
            input=composed_messages,
            model_kwargs=composed_model_kwargs,
            model_type=self.model_type,
        )
        return self._post_call(completion)

    async def acall(
        self,
        input: str,
        prompt_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
    ) -> GeneratorOutputType:
        r"""Async call the model with the input and model_kwargs.
        Note: watch out for the rate limit and the timeout.
        """
        composed_messages, composed_model_kwargs = self._pre_call(
            input, prompt_kwargs, model_kwargs
        )

        completion = await self.model_client.acall(
            input=composed_messages,
            model_kwargs=composed_model_kwargs,
            model_type=self.model_type,
        )
        return self._post_call(completion)
