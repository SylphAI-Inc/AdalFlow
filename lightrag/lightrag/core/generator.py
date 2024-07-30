"""Generator is a user-facing orchestration component with a simple and unified interface for LLM prediction.

It is a pipeline that consists of three subcomponents."""

from typing import Any, Dict, Optional, Union, Callable
from copy import deepcopy
import logging

from lightrag.core.types import (
    ModelType,
    GeneratorOutput,
    GeneratorOutputType,
)
from lightrag.core.component import Component

# Avoid circular import
# if TYPE_CHECKING:
from lightrag.optim.parameter import Parameter, GradientContext

from lightrag.core.prompt_builder import Prompt
from lightrag.core.functional import compose_model_kwargs
from lightrag.core.model_client import ModelClient
from lightrag.core.default_prompt_template import DEFAULT_LIGHTRAG_SYSTEM_PROMPT
from lightrag.optim.text_grad.function import BackwardContext, GradFunction

from lightrag.optim.text_grad.backend_engine_prompt import (
    FEEDBACK_ENGINE_TEMPLATE,
    CONVERSATION_TEMPLATE,
    CONVERSATION_START_INSTRUCTION_BASE,
    CONVERSATION_START_INSTRUCTION_CHAIN,
    EVALUATE_VARIABLE_INSTRUCTION,
    OBJECTIVE_INSTRUCTION_BASE,
    OBJECTIVE_INSTRUCTION_CHAIN,
)

log = logging.getLogger(__name__)


def _convert_prompt_kwargs_to_str(prompt_kwargs: Dict) -> Dict[str, str]:
    r"""Convert the prompt_kwargs to a dictionary with string values."""
    prompt_kwargs_str: Dict[str, str] = {}

    for key, p in prompt_kwargs.items():

        if isinstance(p, Parameter):

            prompt_kwargs_str[key] = p.data
        else:
            prompt_kwargs_str[key] = p
    return prompt_kwargs_str


class Generator(Component, GradFunction):
    __doc__ = """An user-facing orchestration component for LLM prediction.

    It is also a GradFunction that can be used for backpropagation through the LLM model.

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
        # trainable_params: Optional[List[str]] = [],
        name: Optional[str] = None,
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
            raise TypeError(
                f"{type(self).__name__} requires a ModelClient instance for model_client, please pass it as OpenAIClient() or GroqAPIClient() for example."
            )

        template = template or DEFAULT_LIGHTRAG_SYSTEM_PROMPT
        try:
            prompt_kwargs = deepcopy(prompt_kwargs)
        except Exception as e:
            log.warning(f"Error copying the prompt_kwargs: {e}")
            prompt_kwargs = prompt_kwargs

        super().__init__(
            # model_kwargs=model_kwargs,
            # template=template,
            # prompt_kwargs=prompt_kwargs,
            # trainable_params=trainable_params,
        )
        self.name = name or self.__class__.__name__

        self._init_prompt(template, prompt_kwargs)

        self.model_kwargs = model_kwargs.copy()
        # init the model client
        self.model_client = model_client

        self.output_processors = output_processors

        # add trainable_params to generator
        # prompt_variables = self.prompt.get_prompt_variables()
        for key, p in prompt_kwargs.items():
            if isinstance(p, Parameter):
                setattr(self, key, p)
        # self._trainable_params: List[str] = []
        # for param in trainable_params:
        #     if param not in prompt_variables:
        #         raise ValueError(
        #             f"trainable_params: {param} not found in the prompt_variables: {prompt_variables}"
        #         )
        #     # Create a Parameter object and assign it as an attribute with the same name as the value of param
        #     default_value = self.prompt_kwargs.get(param, None)
        #     setattr(self, param, Parameter[Union[str, None]](data=default_value))
        #     self._trainable_params.append(param)
        # end of trainable parameters
        self.backward_engine: "BackwardEngine" = None
        log.info(f"Generator {self.name} initialized.")
        #  to support better testing on the parts beside of the model call
        self.mock_output: bool = False
        self.mock_output_data: str = "mock data"
        self.data_map_func: Callable = None
        self.set_data_map_func()

    def set_mock_output(
        self, mock_output: bool = True, mock_output_data: str = "mock data"
    ):
        self.mock_output = mock_output
        self.mock_output_data = mock_output_data

    def reset_mock_output(self):
        self.mock_output = False
        self.mock_output_data = "mock data"

    def _init_prompt(self, template: str, prompt_kwargs: Dict):
        r"""Initialize the prompt with the template and prompt_kwargs."""
        self.template = template
        self.prompt_kwargs = prompt_kwargs
        self.prompt_kwargs_str = _convert_prompt_kwargs_to_str(prompt_kwargs)
        self.prompt = Prompt(template=template, prompt_kwargs=self.prompt_kwargs_str)

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
        prompt_kwargs_str = _convert_prompt_kwargs_to_str(kwargs)
        return self.prompt.print_prompt(**prompt_kwargs_str)

    def _extra_repr(self) -> str:
        s = f"model_kwargs={self.model_kwargs}, model_type={self.model_type}"
        return s

    def _post_call(self, completion: Any) -> GeneratorOutputType:
        r"""Get string completion and process it with the output_processors."""
        try:
            response = self.model_client.parse_chat_completion(completion)
        except Exception as e:
            log.error(f"Error parsing the completion {completion}: {e}")
            return GeneratorOutput(raw_response=str(completion), error=str(e))

        # the output processors operate on the str, the raw_response field.
        output: GeneratorOutputType = GeneratorOutput(raw_response=str(response))

        if self.output_processors:
            try:
                response = self.output_processors(response)
                output.data = response
            except Exception as e:
                log.error(f"Error processing the output processors: {e}")
                output.error = str(e)
        else:  # default to string output
            output.data = response

        return output

    def _pre_call(self, prompt_kwargs: Dict, model_kwargs: Dict) -> Dict[str, Any]:
        r"""Prepare the input, prompt_kwargs, model_kwargs for the model call."""
        # 1. render the prompt from the template
        prompt_kwargs_str = _convert_prompt_kwargs_to_str(prompt_kwargs)
        prompt_str = self.prompt.call(**prompt_kwargs_str).strip()

        # 2. combine the model_kwargs with the default model_kwargs
        composed_model_kwargs = self._compose_model_kwargs(**model_kwargs)

        # 3. convert app's inputs to api inputs
        api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
            input=prompt_str,
            model_kwargs=composed_model_kwargs,
            model_type=self.model_type,
        )
        return api_kwargs

    def set_backward_engine(self, backward_engine: "BackwardEngine" = None):
        if backward_engine is None:
            backward_engine = BackwardEngine(
                model_client=self.model_client,
                model_kwargs=self.model_kwargs,
            )
            if self.mock_output:
                backward_engine.set_mock_output()
        print(f"Setting backward engine: {backward_engine}")
        self.backward_engine = backward_engine
        # super().set_backward_engine(backward_engine)
        print(f"Backward engine set: {self.backward_engine}")

    def set_data_map_func(self, map_func: Callable = None):
        def default_map_func(data: "GeneratorOutputType") -> str:
            return (
                data.data
                if data.data
                else self.failure_message_to_backward_engine(data)
            )

        self.data_map_func = map_func or default_map_func

        log.debug(f"Data map function set: {self.data_map_func}")

    # NOTE: when training is true, we use forward instead of call
    def forward(
        self,
        prompt_kwargs: Optional[Dict] = {},  # the input need to be passed to the prompt
        model_kwargs: Optional[Dict] = {},
    ) -> "Parameter":
        # 1. call the model
        output: GeneratorOutputType = None
        if self.mock_output:
            output = GeneratorOutput(data=self.mock_output_data)
        else:
            output = self.call(prompt_kwargs, model_kwargs)
        # 2. Generate a Parameter object from the output
        # parameter_data = None
        # if output.data is None:
        #     parameter_data = (
        #         f"raw response: {output.raw_response}" + "error: " + output.error
        #     )

        # else:
        #     parameter_data = output.data
        combined_prompt_kwargs = compose_model_kwargs(self.prompt_kwargs, prompt_kwargs)
        if self.data_map_func is None:
            self.set_data_map_func()
        response: Parameter = Parameter(
            data=self.data_map_func(output),
            alias=self.name + "_output",
            predecessors=[
                p for p in combined_prompt_kwargs.values() if isinstance(p, Parameter)
            ],
            role_desc=f"response from generator {self.name}",
            raw_response=output.raw_response,
        )
        if not self.backward_engine:
            print("Setting default backward engine")
            self.set_backward_engine()
            print(f"Backward engine: {self.backward_engine}")

        response.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward,
                backward_engine=self.backward_engine,
                response=response,
                prompt_kwargs=combined_prompt_kwargs,
                prompt_str=self.print_prompt(**combined_prompt_kwargs),
            )
        )
        return response

    # == pytorch custom autograd function ==
    def backward(
        self,
        response: Parameter,
        prompt_kwargs: Dict,
        backward_engine: "Generator",
        prompt_str: str,
    ) -> Parameter:

        log.info(f"Generator: Backward: {response}")

        children_params = response.predecessors
        is_chain = True
        if response.get_gradient_and_context_text().strip() == "":
            log.info(f"Generator: Backward: No gradient found for {response}.")
        # Compute all predecessors's gradients based on the current response' note.
        for pred in children_params:
            if not pred.requires_opt:
                log.debug(
                    f"EvalFnToTextLoss: Skipping {pred} as it does not require optimization."
                )
                continue
            self._backward_through_one_predecessor(
                pred,
                response,
                prompt_kwargs,
                backward_engine,
                prompt_str,
                is_chain,
            )

    @staticmethod
    def _backward_through_one_predecessor(
        pred: Parameter,
        response: Parameter,
        prompt_kwargs: Dict[str, str],
        backward_engine: "BackwardEngine",
        prompt_str: str,
        is_chain: bool = False,
    ):
        if not pred.requires_opt:
            log.debug(
                f"Generator: Skipping {pred} as it does not require optimization."
            )
            return
        log.debug(f"Generator: Backward through {pred}, is_chain: {is_chain}")

        instruction_str, objective_str = None, None

        # 1. Generate the conversation string

        conversation_prompt_kwargs = {
            "llm_prompt": prompt_str,
            "response_value": response.raw_response or response.data,
        }

        conversation_str = Prompt(  # takes prompt_kwargs and response_value
            template=CONVERSATION_TEMPLATE, prompt_kwargs=conversation_prompt_kwargs
        )()
        log.info(f"Conversation str: {conversation_str}")

        conv_ins_template = CONVERSATION_START_INSTRUCTION_BASE
        obj_ins_template = OBJECTIVE_INSTRUCTION_BASE
        if is_chain:
            conv_ins_template = CONVERSATION_START_INSTRUCTION_CHAIN
            obj_ins_template = OBJECTIVE_INSTRUCTION_CHAIN

        instruction_str = Prompt(
            template=conv_ins_template,
            prompt_kwargs={
                "variable_desc": pred.role_desc,
                "conversation_str": conversation_str,
            },
        )()
        log.info(f"Conversation start instruction base str: {instruction_str}")
        objective_str = Prompt(
            template=obj_ins_template,
            prompt_kwargs={
                "response_desc": response.role_desc,
                "response_gradient": response.get_gradient_text(),
            },
        )()
        evaluation_variable_instruction_str = Prompt(
            template=EVALUATE_VARIABLE_INSTRUCTION,
            prompt_kwargs={
                "variable_desc": pred.role_desc,
                "variable_short": pred.raw_response or pred.data,
            },
        )()

        log.info(
            f"Evaluation variable instruction str: {evaluation_variable_instruction_str}"
        )
        backward_engine_prompt_kwargs = {
            "conversation_sec": instruction_str,
            "objective_instruction_sec": objective_str,
            "evaluate_variable_instruction_sec": evaluation_variable_instruction_str,
        }

        gradient_output: GeneratorOutput = backward_engine(
            prompt_kwargs=backward_engine_prompt_kwargs
        )
        # USE this to trace each node's input and output, all nodes can be visualized
        log.info(
            f"Generator Backward Engine Prompt: {backward_engine.print_prompt( **backward_engine_prompt_kwargs)}"
        )
        gradient_value = (
            gradient_output.data
            or backward_engine.failure_message_to_optimizer(gradient_output)
        )
        # printc(f"Gradient value: {gradient_value}", color="green")
        log.info(
            f"Generator Gradient value: {gradient_value}, raw response: {gradient_output.raw_response}"
        )
        # TODO: make it a debug feature
        prompt_str = backward_engine.print_prompt(**backward_engine_prompt_kwargs)

        var_gradient = Parameter(
            alias=f"{response.alias}_to_{pred.alias}_grad",
            gradient_prompt=prompt_str,  # trace the prompt
            # raw_response=gradient_output.raw_response,
            data=gradient_value,
            requires_opt=True,
            role_desc=f"feedback to {pred.role_desc}",
        )
        # add the graidents to the variable
        pred.gradients.add(var_gradient)
        # save the gradient context
        # TODO: add an id for each parameter
        pred.gradients_context[var_gradient] = GradientContext(
            context=conversation_str,
            response_desc=response.role_desc,
            variable_desc=pred.role_desc,
        )

    def call(
        self,
        prompt_kwargs: Optional[Dict] = {},  # the input need to be passed to the prompt
        model_kwargs: Optional[Dict] = {},
    ) -> GeneratorOutputType:
        r"""
        Call the model_client by formatting prompt from the prompt_kwargs,
        and passing the combined model_kwargs to the model client.
        """
        if self.mock_output:
            return GeneratorOutput(data=self.mock_output_data)

        log.debug(f"prompt_kwargs: {prompt_kwargs}")
        log.debug(f"model_kwargs: {model_kwargs}")

        api_kwargs = self._pre_call(prompt_kwargs, model_kwargs)
        log.debug(f"api_kwargs: {api_kwargs}")
        output: GeneratorOutputType = None
        # call the model client
        completion = None
        try:
            completion = self.model_client.call(
                api_kwargs=api_kwargs, model_type=self.model_type
            )
        except Exception as e:
            log.error(f"Error calling the model: {e}")
            output = GeneratorOutput(error=str(e))
        # process the completion
        if completion:
            try:
                output = self._post_call(completion)

            except Exception as e:
                log.error(f"Error processing the output: {e}")
                output = GeneratorOutput(raw_response=str(completion), error=str(e))

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

    def __call__(self, *args, **kwargs) -> Union[GeneratorOutputType, Any]:
        if self.training:
            print("Training mode")
            return self.forward(*args, **kwargs)
        else:
            print("Inference mode")
            return self.call(*args, **kwargs)

    def _extra_repr(self) -> str:
        s = f"model_kwargs={self.model_kwargs}, "
        return s

    @staticmethod
    def failure_message_to_backward_engine(
        gradient_response: GeneratorOutput,
    ) -> Optional[str]:
        response_value = None
        if gradient_response.error or not gradient_response.data:
            response_value = f"Error: {gradient_response.error}, Raw response: {gradient_response.raw_response}"
        return response_value


class BackwardEngine(Generator):  # it is a generator with defaule template

    def __init__(self, **kwargs):
        if "template" not in kwargs:
            kwargs["template"] = FEEDBACK_ENGINE_TEMPLATE
        super().__init__(**kwargs)

    @staticmethod
    def failure_message_to_optimizer(
        gradient_response: GeneratorOutput,
    ) -> Optional[str]:
        gradient_value_data = None
        if gradient_response.error or not gradient_response.data:
            gradient_value_data = f"The backward engine failed to compute the gradient. Raw response: {gradient_response.raw_response}, Error: {gradient_response.error}"

        return gradient_value_data


if __name__ == "__main__":
    # test the generator with backward engine
    from lightrag.core.model_client import ModelClient

    # setup_env()
    # llama3_model = {
    #     "model_client": GroqAPIClient(),
    #     "model_kwargs": {
    #         "model": "llama-3.1-8b-instant",
    #     },
    # }
    mock_model = {
        "model_client": ModelClient(),
        "model_kwargs": {
            "model": "mock",
        },
    }
