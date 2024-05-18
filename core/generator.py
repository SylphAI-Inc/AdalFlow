from typing import Any, Dict, List, Optional, Generic, TypeVar, Tuple
from core.data_classes import ModelType
from core.component import Component
from core.prompt_builder import Prompt
from core.functional import compose_model_kwargs
from core.api_client import APIClient
from core.default_prompt_template import DEFAULT_LIGHTRAG_PROMPT

# TODO: special delimiters for different sections

GeneratorInput = str
GeneratorOutput = TypeVar("GeneratorOutput")


# TODO: investigate more on the type checking
class Generator(Generic[GeneratorOutput], Component):
    """
    An orchestrator component that combines the Prompt and the API client to generate text from a prompt.
    Additionally, it allows you to pass the output_processors to further parse the output from the model. Thus the arguments are almost a combination of that of Prompt and APIClient.

    It takes the user query as input in string format, and returns the response or processed response.
    """

    model_type: ModelType = ModelType.LLM
    model_client: APIClient
    prompt: Prompt
    output_processors: Optional[Component]

    def __init__(
        self,
        *,
        model_client: APIClient,
        model_kwargs: Optional[Dict] = {},
        # args for the prompt
        template: str = DEFAULT_LIGHTRAG_PROMPT,
        preset_prompt_kwargs: Optional[Dict] = None,  # manage the prompt kwargs
        output_processors: Optional[Component] = None,
    ) -> None:
        r"""The default prompt is set to the DEFAULT_LIGHTRAG_PROMPT. It has the following variables:
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
        self.prompt = Prompt(
            template=template, preset_prompt_kwargs=preset_prompt_kwargs
        )
        self.output_processors = output_processors

    def train(self, *args, **kwargs):
        pass

    def _compose_lm_input_chat(self, **kwargs: Any) -> List[Dict]:
        """
        Forms the final messages to LLM chat model.

        example:
        {
            "role": "system",

        }


        """
        if not hasattr(self, "prompt") or not self.prompt:
            raise ValueError(
                f"{type(self).__name__} requires a 'prompt' to be set before calling the model."
            )
        current_role: str = kwargs.get("role", "system")
        # TODO: clean up
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

    # def compose_prompt_kwargs(self, **kwargs) -> Dict:
    #     composed_kwargs = (
    #         self.preset_prompt_kwargs.copy() if self.preset_prompt_kwargs else {}
    #     )
    #     composed_kwargs.update(kwargs)
    #     return composed_kwargs

    def print_prompt(self, **kwargs) -> str:
        self.prompt.print_prompt(**kwargs)

    # TODO: move this to output_processors
    def parse_completion(self, completion: Any) -> str:
        """
        Parse the completion to a structure your sytem standarizes. (here is str)
        # TODO: standardize the completion
        """
        return completion.choices[0].message.content

    def extra_repr(self) -> str:
        s = f"model_kwargs={self.model_kwargs} "
        return s

    def _pre_call(
        self,
        input: str,
        prompt_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
    ) -> Tuple[List[Dict], Dict]:
        r"""Compose the input and model_kwargs before calling the model."""
        composed_model_kwargs = self.update_default_model_kwargs(**model_kwargs)
        # add the input to the prompt kwargs
        prompt_kwargs["query_str"] = input
        prompt_str = self.prompt(**prompt_kwargs)
        # TODO: the message might be api specific
        composed_messages = [{"role": "system", "content": prompt_str}]
        # composed_messages = self.compose_model_input(prompt_str=prompt_str)
        return composed_messages, composed_model_kwargs

    def _post_call(self, completion: Any) -> GeneratorOutput:
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
    ) -> GeneratorOutput:
        r"""Call the model with the input and model_kwargs."""
        composed_messages, composed_model_kwargs = self._pre_call(
            input, prompt_kwargs, model_kwargs
        )
        completion = self.model_client.call(
            input=composed_messages,
            model_kwargs=composed_model_kwargs,
            model_type=ModelType.LLM,
        )
        return self._post_call(completion)

    async def acall(
        self,
        input: str,
        prompt_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
    ) -> GeneratorOutput:
        r"""Async call the model with the input and model_kwargs.
        Note: watch out for the rate limit and the timeout.
        """
        composed_messages, composed_model_kwargs = self._pre_call(
            input, prompt_kwargs, model_kwargs
        )
        completion = await self.model_client.acall(
            input=composed_messages,
            model_kwargs=composed_model_kwargs,
            model_type=ModelType.LLM,
        )
        return self._post_call(completion)
