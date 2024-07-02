"""Generator is a user-facing orchestration component with a simple and unified interface for LLM prediction. 

It is a pipeline that consists of three subcomponents."""
from typing import Any, Dict, List, Optional, Union
from copy import deepcopy
import logging

from lightrag.core.types import (
    ModelType,
    GeneratorOutput,
    GeneratorOutputType,
)
from lightrag.core.component import Component
from lightrag.core.parameter import Parameter
from lightrag.core.prompt_builder import Prompt
from lightrag.core.model_client import ModelClient
from lightrag.core.default_prompt_template import DEFAULT_LIGHTRAG_SYSTEM_PROMPT


log = logging.getLogger(__name__)


class Generator(Component):
    """
    An user-facing orchestration component for LLM prediction.

    By orchestrating the following three components along with their required arguments,
    it enables any LLM prediction with required task output format.
    - Prompt
    - Model client
    - Output processors

    Args:
        model_client (ModelClient): The model client to use for the generator.
        model_kwargs (Dict[str, Any], optional): The model kwargs to pass to the model client. Defaults to {}. Please refer to :ref:`ModelClient<components-model_client>` for the details on how to set the model_kwargs for your specific model if it is from our library.
        template (Optional[str], optional): The template for the prompt.  Defaults to :ref:`DEFAULT_LIGHTRAG_SYSTEM_PROMPT<core-default_prompt_template>`.
        prompt_kwargs (Optional[Dict], optional): The preset prompt kwargs to fill in the variables in the prompt. Defaults to None.
        output_processors (Optional[Component], optional):  The output processors after model call. It can be a single component or a chained component via ``Sequential``. Defaults to None.
        trainable_params (Optional[List[str]], optional): The list of trainable parameters. Defaults to [].

    Note:
        The output_processors will be applied to the string output of the model completion. And the result will be stored in the data field of the output. And we encourage you to only use it to parse the response to data format you will use later.
    """

    model_type: ModelType = ModelType.LLM
    model_client: ModelClient  # for better type checking

    def __init__(
        self,
        *,
        # args for the model
        model_client: ModelClient,  # will be intialized in the main script
        model_kwargs: Dict[str, Any] = {},
        # args for the prompt
        template: Optional[str] = None,
        prompt_kwargs: Optional[Dict] = {},
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
        You can preset the prompt kwargs to fill in the variables in the prompt using prompt_kwargs.
        But you can replace the prompt and set any variables you want and use the prompt_kwargs to fill in the variables.
        """

        if not isinstance(model_client, ModelClient):
            raise ValueError(
                f"{type(self).__name__} requires a ModelClient instance for model_client, please pass it as OpenAIClient() or GroqAPIClient() for example."
            )

        template = template or DEFAULT_LIGHTRAG_SYSTEM_PROMPT
        try:
            prompt_kwargs = deepcopy(prompt_kwargs)
        except Exception as e:
            log.warning(f"Error copying the prompt_kwargs: {e}")
            prompt_kwargs = prompt_kwargs

        super().__init__(
            model_kwargs=model_kwargs,
            template=template,
            prompt_kwargs=prompt_kwargs,
            trainable_params=trainable_params,
        )

        self._init_prompt(template, prompt_kwargs)

        self.model_kwargs = model_kwargs.copy()
        # init the model client
        self.model_client = model_client

        self.output_processors = output_processors

        # add trainable_params to generator
        prompt_variables = self.prompt.get_prompt_variables()
        self._trainable_params: List[str] = []
        for param in trainable_params:
            if param not in prompt_variables:
                raise ValueError(
                    f"trainable_params: {param} not found in the prompt_variables: {prompt_variables}"
                )
            # Create a Parameter object and assign it as an attribute with the same name as the value of param
            default_value = self.prompt_kwargs.get(param, None)
            setattr(self, param, Parameter[Union[str, None]](data=default_value))
            self._trainable_params.append(param)
        # end of trainable parameters

    def _init_prompt(self, template: str, prompt_kwargs: Dict):
        r"""Initialize the prompt with the template and prompt_kwargs."""
        self.template = template
        self.prompt_kwargs = prompt_kwargs
        self.prompt = Prompt(template=template, prompt_kwargs=prompt_kwargs)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Generator":
        r"""Create a Generator instance from the config dictionary.

        Example:

        .. code-block:: python

            config = {
                        "model_client": {
                            "component_name": "OpenAIClient",
                            "component_config": {}
                        },
                        "model_kwargs": {"model": "gpt-3.5-turbo", "temperature": 0}
                    }
            generator = Generator.from_config(config)
        """
        # create init_kwargs from the config
        assert "model_client" in config, "model_client is required in the config"
        return super().from_config(config)

    # def _compose_lm_input_non_chat(self, **kwargs: Any) -> str:
    #     """
    #     This combines the default lm input using Prompt, and the passed input. history, steps, etc.
    #     It builds the final chat input to the model.

    #     As
    #     """
    #     prompt_text = self.prompt.call(**kwargs)
    #     return prompt_text

    def _compose_model_kwargs(self, **model_kwargs) -> Dict:
        r"""
        The model configuration exclude the input itself.
        Combine the default model, model_kwargs with the passed model_kwargs.
        Example:
        model_kwargs = {"temperature": 0.5, "model": "gpt-3.5-turbo"}
        self.model_kwargs = {"model": "gpt-3.5-turbo"}
        combine_kwargs(model_kwargs) => {"temperature": 0.5, "model": "gpt-3.5-turbo"}

        """
        combined_model_kwargs = self.model_kwargs.copy()

        if model_kwargs:
            combined_model_kwargs.update(model_kwargs)
        return combined_model_kwargs

    def print_prompt(self, **kwargs) -> str:
        self.prompt.print_prompt(**kwargs)

    def _extra_repr(self) -> str:
        s = f"model_kwargs={self.model_kwargs}, model_type={self.model_type}"
        return s

    def _post_call(self, completion: Any) -> GeneratorOutputType:
        r"""Get string completion and process it with the output_processors."""
        try:
            response = self.model_client.parse_chat_completion(completion)
        except Exception as e:
            log.error(f"Error parsing the completion: {e}")
            # response = str(completion)
            return GeneratorOutput(raw_response=str(completion), error=str(e))

        # the output processors operate on the str, the raw_response field.
        output: GeneratorOutputType = GeneratorOutput(raw_response=response)

        # TODO: this output processing patterns need to be more clear
        response = deepcopy(response)
        if self.output_processors:
            try:
                response = self.output_processors(response)
                output.data = response
            except Exception as e:
                log.error(f"Error processing the output: {e}")
                output.error = str(e)
        else:  # default to string output
            output.data = response

        return output

    def _pre_call(self, prompt_kwargs: Dict, model_kwargs: Dict) -> Dict[str, Any]:
        r"""Prepare the input, prompt_kwargs, model_kwargs for the model call."""
        # 1. render the system prompt from the template
        system_prompt_str = self.prompt.call(**prompt_kwargs).strip()

        # 2. combine the model_kwargs with the default model_kwargs
        composed_model_kwargs = self._compose_model_kwargs(**model_kwargs)

        # 3. convert app's inputs to api inputs
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
        r"""
        Call the model_client by formatting prompt from the prompt_kwargs,
        and passing the combined model_kwargs to the model client.
        """

        if self.training:
            # add the parameters to the prompt_kwargs
            # convert attributes to prompt_kwargs
            trained_prompt_kwargs = {
                param: getattr(self, param).data for param in self.state_dict()
            }
            prompt_kwargs.update(trained_prompt_kwargs)

        log.info(f"prompt_kwargs: {prompt_kwargs}")
        log.info(f"model_kwargs: {model_kwargs}")

        api_kwargs = self._pre_call(prompt_kwargs, model_kwargs)
        output: GeneratorOutputType = None
        # call the model client
        try:
            completion = self.model_client.call(
                api_kwargs=api_kwargs, model_type=self.model_type
            )
            output = self._post_call(completion)
        except Exception as e:
            log.error(f"Error calling the model: {e}")
            output = GeneratorOutput(error=str(e))

        log.info(f"output: {output}")
        return output

    # TODO: training is not supported in async call yet
    async def acall(
        self,
        prompt_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
    ) -> GeneratorOutputType:
        r"""Async call the model with the input and model_kwargs.

        :warning::
            Training is not supported in async call yet.
        """
        log.info(f"prompt_kwargs: {prompt_kwargs}")
        log.info(f"model_kwargs: {model_kwargs}")

        api_kwargs = self._pre_call(prompt_kwargs, model_kwargs)
        completion = await self.model_client.acall(
            api_kwargs=api_kwargs, model_type=self.model_type
        )
        output = self._post_call(completion)
        log.info(f"output: {output}")
        return output

    def _extra_repr(self) -> str:
        s = f"model_kwargs={self.model_kwargs}, "
        return s
