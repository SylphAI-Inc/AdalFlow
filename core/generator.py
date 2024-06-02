from typing import Any, Dict, List, Optional
from copy import deepcopy
import logging

from core.data_classes import ModelType, GeneratorOutput
from core.component import Component
from core.parameter import Parameter
from core.prompt_builder import Prompt
from core.functional import compose_model_kwargs
from core.api_client import APIClient
from core.default_prompt_template import DEFAULT_LIGHTRAG_SYSTEM_PROMPT


GeneratorInputType = str
GeneratorOutputType = GeneratorOutput

logger = logging.getLogger(__name__)


# NOTE: currently generator cannot be used in Sequential due to specialized output data type
# TODO: generator should track its failed calls so that users can review them, and save the failed calls to a file
class Generator(Component):
    """
    An user-facing orchestration component for LLM prediction.

    By orchestrating the following three components along with their required arguments,
    it enables any LLM prediction with required task output format.
    - Prompt
    - Model client
    - Output processors
    """

    model_type: ModelType = ModelType.LLM
    model_client: APIClient  # for better type checking

    def __init__(
        self,
        *,
        # args for the model
        model_client: APIClient,  # will be intialized in the main script
        model_kwargs: Dict[str, Any] = {},
        # args for the prompt
        template: str = DEFAULT_LIGHTRAG_SYSTEM_PROMPT,
        preset_prompt_kwargs: Optional[Dict] = None,
        # args for the output processing
        output_processors: Optional[Component] = None,
        # args for the trainable parameters
        trainable_params: Optional[List[str]] = [],
    ) -> None:
        r"""The default prompt is set to the DEFAULT_LIGHTRAG_SYSTEM_PROMPT. It has the following variables:
        - task_desc_str
        - tools_str
        - example_str
        - chat_history_str
        - context_str
        - steps_str
        You can preset the prompt kwargs to fill in the variables in the prompt using preset_prompt_kwargs.
        But you can replace the prompt and set any variables you want and use the preset_prompt_kwargs to fill in the variables.
        """
        super().__init__()

        self._init_prompt(template, preset_prompt_kwargs)

        self.model_kwargs = model_kwargs
        # init the model client
        self.model_client = model_client

        self.output_processors = output_processors

        # add trainable_params to generator
        prompt_variables = self.system_prompt.get_prompt_variables()
        self._trainable_params: List[str] = []
        for param in trainable_params:
            if param not in prompt_variables:
                raise ValueError(
                    f"trainable_params: {param} not found in the prompt_variables: {prompt_variables}"
                )
            # Create a Parameter object and assign it as an attribute with the same name as the value of param
            default_value = self.preset_prompt_kwargs.get(param, None)
            setattr(self, param, Parameter(data=default_value))
            self._trainable_params.append(param)
        # end of trainable parameters

    def _init_prompt(self, template: str, preset_prompt_kwargs: Dict):
        r"""Initialize the prompt with the template and preset_prompt_kwargs."""
        self.template = template
        self.preset_prompt_kwargs = preset_prompt_kwargs
        self.system_prompt = Prompt(
            template=template, preset_prompt_kwargs=preset_prompt_kwargs
        )
        # return Prompt(template=template, preset_prompt_kwargs=preset_prompt_kwargs)

    # def _compose_lm_input_non_chat(self, **kwargs: Any) -> str:
    #     """
    #     This combines the default lm input using Prompt, and the passed input. history, steps, etc.
    #     It builds the final chat input to the model.

    #     As
    #     """
    #     prompt_text = self.system_prompt.call(**kwargs)
    #     return prompt_text

    def update_default_model_kwargs(self, **model_kwargs) -> Dict:
        r"""
        The model configuration exclude the input itself.
        Combine the default model, model_kwargs with the passed model_kwargs.
        Example:
        model_kwargs = {"temperature": 0.5, "model": "gpt-3.5-turbo"}
        self.model_kwargs = {"model": "gpt-3.5-turbo"}
        combine_kwargs(model_kwargs) => {"temperature": 0.5, "model": "gpt-3.5-turbo"}

        """
        return compose_model_kwargs(self.model_kwargs, model_kwargs)

    def print_prompt(self, **kwargs) -> str:
        self.system_prompt.print_prompt(**kwargs)

    def _extra_repr(self) -> str:
        s = f"model_kwargs={self.model_kwargs}, model_type={self.model_type}"
        return s

    def _post_call(self, completion: Any) -> GeneratorOutputType:
        r"""Parse the completion and process the output."""
        try:
            response = self.model_client.parse_chat_completion(completion)
        except Exception as e:
            response = str(completion)
            return GeneratorOutput(raw_response=response, error_message=str(e))

        output: GeneratorOutputType = GeneratorOutput(raw_response=response)
        response = deepcopy(response)
        if self.output_processors:
            try:
                response = self.output_processors(response)
            except Exception as e:
                output.error_message = str(e)
        output.data = response
        return output

    def _pre_call(self, prompt_kwargs: Dict, model_kwargs: Dict) -> Dict[str, Any]:
        r"""Prepare the input, prompt_kwargs, model_kwargs for the model call."""
        # step 1: render the system prompt
        system_prompt_str = self.system_prompt.call(**prompt_kwargs).strip()

        # step 2: combine the model_kwargs with the default model_kwargs
        composed_model_kwargs = self.update_default_model_kwargs(**model_kwargs)

        # step 3: use model_client.combined_input_and_model_kwargs to get the api_kwargs
        api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
            input=system_prompt_str,
            model_kwargs=composed_model_kwargs,
            model_type=self.model_type,
        )
        return api_kwargs

    def call(
        self,
        prompt_kwargs: Optional[Dict] = {},  # the input need to be passed to the prompt
        model_kwargs: Optional[Dict] = {},
    ) -> GeneratorOutputType:
        r"""Call the model with the input(user_query) and model_kwargs."""

        if self.training:
            # add the parameters to the prompt_kwargs
            # convert attributes to prompt_kwargs
            trained_prompt_kwargs = {
                param: getattr(self, param).data for param in self.state_dict()
            }
            prompt_kwargs.update(trained_prompt_kwargs)

        logger.info(f"prompt_kwargs: {prompt_kwargs}")
        logger.info(f"model_kwargs: {model_kwargs}")

        api_kwargs = self._pre_call(prompt_kwargs, model_kwargs)
        # print(f"api_kwargs: {api_kwargs}")
        completion = self.model_client.call(
            api_kwargs=api_kwargs, model_type=self.model_type
        )
        output = self._post_call(completion)

        logger.info(f"output: {output}")
        return output

    async def acall(
        self,
        prompt_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
    ) -> GeneratorOutputType:
        r"""Async call the model with the input and model_kwargs.
        Note: watch out for the rate limit and the timeout.
        """
        api_kwargs = self._pre_call(prompt_kwargs, model_kwargs)
        completion = await self.model_client.acall(
            api_kwargs=api_kwargs, model_type=self.model_type
        )
        output = self._post_call(completion)
        return output
