"""Generator is a user-facing orchestration component with a simple and unified interface for LLM prediction.

It is a pipeline that consists of three subcomponents."""

import json
import re
import time
from pathlib import Path

from typing import Any, Dict, Optional, Union, Callable, Tuple, List, AsyncGenerator
from collections.abc import AsyncIterable
import logging
from dataclasses import dataclass, field

from openai.types.responses import ResponseCompletedEvent
from adalflow.core.tokenizer import Tokenizer


from adalflow.core.types import (
    ModelType,
    GeneratorOutput,
    GeneratorOutputType,
)
from adalflow.core.component import Component, DataComponent
from adalflow.optim.grad_component import GradComponent
from adalflow.core.base_data_class import DataClass


from adalflow.optim.parameter import (
    Parameter,
    OutputParameter,
)
from adalflow.optim.gradient import GradientContext, Gradient
from adalflow.optim.types import ParameterType

from adalflow.core.prompt_builder import Prompt
from adalflow.core.functional import compose_model_kwargs
from adalflow.core.model_client import ModelClient
from adalflow.core.default_prompt_template import DEFAULT_ADALFLOW_SYSTEM_PROMPT
from adalflow.optim.function import BackwardContext
from adalflow.utils.cache import CachedEngine
from adalflow.tracing.callback_manager import CallbackManager
from adalflow.tracing import generator_span
from adalflow.utils.global_config import get_adalflow_default_root_path
from adalflow.core.string_parser import JsonParser

from adalflow.optim.text_grad.backend_engine_prompt import (
    FEEDBACK_ENGINE_TEMPLATE,
    LLM_CONVERSATION_TEMPLATE,
    ALL_PRED_INFO,
    OUTPUT_INSTRUCTION,
    VARIABLE_AND_PEERS_INFO,
    CONVERSATION_START_INSTRUCTION_CHAIN,
    OBJECTIVE_INSTRUCTION_BASE,
    OBJECTIVE_INSTRUCTION_CHAIN,
)

__all__ = ["Generator", "BackwardEngine", "create_teacher_generator"]


log = logging.getLogger(__name__)


PromptArgType = Dict[str, Union[str, Parameter]]


@dataclass
class BackwardPassSetup(DataClass):
    all_pred_at_once: bool = field(
        default=False, metadata={"desc": "Backward all predecessors at once."}
    )
    threshold_score_to_compute_grad_for_errors: float = field(
        default=0.9,
        metadata={"desc": "Threshold score to compute gradient for errors."},
    )
    compute_grad_for_errors_only: bool = field(
        default=True, metadata={"desc": "Compute gradient for errors only."}
    )


# TODO: better debug mode
class Generator(GradComponent, CachedEngine, CallbackManager):
    __doc__ = """An user-facing orchestration component for LLM prediction.

    It is also a GradComponent that can be used for backpropagation through the LLM model.

    By orchestrating the following three components along with their required arguments,
    it enables any LLM prediction with required task output format.
    - Prompt
    - Model client
    - Output processors

    Args:
        model_client (ModelClient): The model client to use for the generator.
        model_kwargs (Dict[str, Any], optional): The model kwargs to pass to the model client. Defaults to {}. Please refer to :ref:`ModelClient<components-model_client>` for the details on how to set the model_kwargs for your specific model if it is from our library.
        model_type (ModelType, optional): The type of the model. Defaults to ModelType.LLM. When using reasoning models which calls different api, you should set it to ModelType.LLM_REASONING.
        template (Optional[str], optional): The template for the prompt.  Defaults to :ref:`DEFAULT_ADALFLOW_SYSTEM_PROMPT<core-default_prompt_template>`.
        prompt_kwargs (Optional[Dict], optional): The preset prompt kwargs to fill in the variables in the prompt. Defaults to None.
        output_processors (Optional[Component], optional):  The output processors after model call. It can be a single component or a chained component via ``Sequential``. Defaults to None.
        trainable_params (Optional[List[str]], optional): The list of trainable parameters. Defaults to [].

    Note:
        1. The output_processors will be applied to the string output of the model completion. And the result will be stored in the data field of the output.
        And we encourage you to only use it to parse the response to data format you will use later.
        2. For structured output, you should avoid using `stream` as the output_processors can only be run after all the data is available.
    """

    model_type: ModelType = ModelType.LLM
    model_client: ModelClient  # for better type checking

    _use_cache: bool = False
    _kwargs: Dict[str, Any] = (
        {}
    )  # to create teacher generator from student TODO: might reaccess this

    backward_pass_setup: BackwardPassSetup = (
        BackwardPassSetup()
    )  # default setup for the backward pass

    def __init__(
        self,
        *,
        # args for the model
        model_client: ModelClient,  # will be intialized in the main script
        model_kwargs: PromptArgType = {},
        model_type: Optional[ModelType] = ModelType.LLM,
        # args for the prompt
        template: Optional[str] = None,
        prompt_kwargs: Optional[Dict] = {},
        # args for the output processing
        output_processors: Optional[DataComponent] = None,
        name: Optional[str] = None,
        # args for the cache
        cache_path: Optional[str] = None,
        use_cache: bool = True,
    ) -> None:
        r"""The default prompt is set to the DEFAULT_ADALFLOW_SYSTEM_PROMPT. It has the following variables:
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
                f"{type(self).__name__} requires a ModelClient instance for model_client, please pass it as OpenAIClient() or GroqAPIClient() for example.\
                    Got {model_client} instead."
            )

        template = template or DEFAULT_ADALFLOW_SYSTEM_PROMPT

        # create the cache path and initialize the cache engine

        self.set_cache_path(
            cache_path, model_client, model_kwargs.get("model", "default")
        )

        CachedEngine.__init__(self, cache_path=self.cache_path)

        Component.__init__(self)
        GradComponent.__init__(self, desc="Generate a response using LLM model.")
        CallbackManager.__init__(self)

        self.name = name or self.__class__.__name__
        self.template = template
        self.prompt_kwargs = prompt_kwargs.copy()

        self.model_kwargs = model_kwargs.copy()
        # init the model client
        self.model_client = model_client
        self.model_type = model_type

        self.output_processors = output_processors

        if output_processors and (not isinstance(output_processors, DataComponent)):
            raise ValueError(
                f"output_processors should be a DataComponent instance, got {type(output_processors)}"
            )

        self.set_parameters(prompt_kwargs)

        # end of trainable parameters
        self.backward_engine: "BackwardEngine" = None
        log.info(f"Generator {self.name} initialized.")
        #  to support better testing on the parts beside of the model call
        self.mock_output: bool = False
        self.mock_output_data: str = "mock data"

        self._use_cache = use_cache

        self._kwargs = {
            "model_client": model_client,
            "model_kwargs": model_kwargs,
            "template": template,
            "prompt_kwargs": prompt_kwargs,
            "output_processors": output_processors,
            "name": name,
            "cache_path": cache_path,
            "use_cache": use_cache,
        }
        self._teacher: Optional["Generator"] = None
        self._trace_api_kwargs: Dict[str, Any] = (
            {}
        )  # used by dynamic computation graph and backpropagation

        self._tokenizer: Tokenizer = Tokenizer()
        self._estimated_token_count: int = 0

    @property
    def use_cache(self):
        return self._use_cache

    
    @property
    def estimated_token_count(self) -> int:
        """Property to access the estimated token count from the last prompt.
        
        Returns:
            int: The estimated token count from the last processed prompt.
                 Returns 0 if no prompt has been processed yet.
        """
        return self._estimated_token_count

    def update_default_backward_pass_setup(self, setup: BackwardPassSetup):
        self.backward_pass_setup = setup

    def set_cache_path(self, cache_path: str, model_client: object, model: str):
        """Set the cache path for the generator."""

        # Construct a valid model string using the client class name and model
        self.model_str = f"{model_client.__class__.__name__}_{model}"

        # Remove any characters that are not allowed in file names (cross-platform)
        # On Windows, characters like `:<>?/\|*` are prohibited.
        self.model_str = re.sub(r"[^a-zA-Z0-9_\-]", "_", self.model_str)

        _cache_path = (
            get_adalflow_default_root_path() if cache_path is None else cache_path
        )

        # Use pathlib to handle paths more safely across OS
        self.cache_path = Path(_cache_path) / f"cache_{self.model_str}.db"

        log.debug(f"Cache path set to: {self.cache_path}")

    def get_cache_path(self) -> str:
        r"""Get the cache path for the generator."""
        return self.cache_path

    @staticmethod
    def _get_default_mapping(
        output: "GeneratorOutput" = None,
    ) -> Tuple[Dict[str, Callable], List[str]]:

        if (
            output.data
            and isinstance(output.data, DataClass)
            and len(output.data.get_output_fields()) > 0
        ):
            output_fields = output.data.get_output_fields()

            output_mapping = {
                f: lambda x, f=f: getattr(x.data, f) for f in output_fields
            }
        elif output.raw_response:
            output_fields = ["raw_response"]
            output_mapping = {f: lambda x, f=f: getattr(x, f) for f in output_fields}
            output_fields = ["Answer"]
            output_mapping["Example"] = output_mapping["raw_response"]
            del output_mapping["raw_response"]

        return output_mapping, output_fields

    def set_mock_output(
        self, mock_output: bool = True, mock_output_data: str = "mock data"
    ):
        self.mock_output = mock_output
        self.mock_output_data = mock_output_data

    def reset_mock_output(self):
        self.mock_output = False
        self.mock_output_data = "mock data"

    def set_parameters(self, prompt_kwargs: PromptArgType):
        r"""Set name for each paramter and set all context for each other.
        Make all parameters attributes to the generator for finding them easily
        for optimizers and other components.
        """
        for key, p in prompt_kwargs.items():
            if isinstance(p, Parameter):
                if not p.name or p.name == "":
                    p.name = key
                peers = [
                    p
                    for k, p in prompt_kwargs.items()
                    if isinstance(p, Parameter) and k != key
                    # and p.param_type == ParameterType.PROMPT
                ]
                p.set_peers(peers)
                setattr(self, key, p)

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

    # TODO: use prompt_kwargs as users are already familiar with it
    def print_prompt(self, **kwargs) -> str:
        prompt = Prompt(template=self.template, prompt_kwargs=self.prompt_kwargs)
        return prompt.print_prompt(**kwargs)

    def get_prompt(self, **kwargs) -> str:
        prompt = Prompt(template=self.template, prompt_kwargs=self.prompt_kwargs)
        return prompt.call(**kwargs)

    def _extra_repr(self) -> str:
        prompt = Prompt(template=self.template, prompt_kwargs=self.prompt_kwargs)
        s = f"model_kwargs={self.model_kwargs}, model_type={self.model_type}, prompt={prompt}"
        return s

    def _post_call(self, completion: Any) -> GeneratorOutput:
        r"""Get string completion and process it with the output_processors."""
        # parse chat completion will only fill the raw_response
        output: GeneratorOutput = self.model_client.parse_chat_completion(completion)
        # save the api response
        output.api_response = completion

        # Now adding the data field to the output
        data = output.raw_response

        # TODO implement support for synchronous iterator in the future
        if self.output_processors:
            if data:
                try:
                    data = self.output_processors(data)
                    output.data = data
                except Exception as e:
                    log.error(f"Error processing the output processors: {e}")
                    output.error = str(e)
        else:
            output.data = data

        return output

    async def _output_processing(
        self, data: AsyncIterable, generator_output: GeneratorOutput
    ) -> AsyncGenerator[Any, None]:
        r"""Create an async generator from the async_iterable returned by model client and yield from them
        apply output processors and store in generator output.
        Consume the raw_response and yield each event.
        Store the final output text in the generator output data field.
        """

        # the raw response of the generator output should be an async iterable
        if not isinstance(data, AsyncIterable):
            raise ValueError("The data is not an async iterable")

        final_output_text = ""

        # iterate over the events in the data and yield each event and store the final output text
        # this assumes that the raw_response of the generator output stores a
        # async iterable of events based on the OpenAI Responses API documentation for streaming
        async for event in data:
            log.debug(f"Raw event: {event!r}")
            yield event

            if ResponseCompletedEvent and isinstance(event, ResponseCompletedEvent):
                resp = event.response
                log.debug(f"Response completed: {event.response.output_text}")
                if getattr(resp, "output_text", None):
                    final_output_text = resp.output_text

        if self.output_processors:
            if final_output_text:
                try:
                    final_output = self.output_processors(final_output_text)
                    generator_output.data = final_output
                except Exception as e:
                    log.error(f"Error processing the output processors: {e}")
                    generator_output.error = str(e)
        else:
            generator_output.data = None

    async def _async_post_call(self, completion: Any) -> GeneratorOutput:
        r"""Get completion and depending on whether the client is streaming from the model client, create a GeneratorOutput or an AsyncGenerator.
        when the client is not streaming, the post call return a GeneratorOutput where the final output after applying the output processors is stored under the data field.
        When the client is streaming, the post call returns an Async Generator which will yield the events under the raw response and then also
        the final Generator Output. The final Generator Output will have under data the final output after applying the output processors.
        """
        # parse chat completion will only fill the raw_response
        output: GeneratorOutput = self.model_client.parse_chat_completion(completion)
        # save the api response
        output.api_response = completion

        log.info(
            f"Response from the Model Client Stream before being processed: {output.raw_response}"
        )

        # Now adding the data field to the output and setting as the raw response
        data = output.raw_response

        # Handle async iterables from OpenAI Agent/Responses API streaming
        # return an async generator
        if isinstance(output.raw_response, AsyncIterable):
            original_raw_response = (
                output.raw_response
            )  # pass in the raw response to the output processing to avoid circular dependency
            output.raw_response = self._output_processing(original_raw_response, output)
        else:
            # return a GeneratorOutput if the raw response is not an async iterable
            # process the model client's final response with the output processors
            log.info(f"Response from the Model Client before being processed: {data}")
            if self.output_processors:
                if data:
                    try:
                        data = self.output_processors(data)
                        output.data = data
                    except Exception as e:
                        log.error(f"Error processing the output processors: {e}")
                        output.error = str(e)
            else:
                output.data = data
            log.info(f"Response from the Model Client after being processed: {data}")

        return output

    def _pre_call(self, prompt_kwargs: Dict, model_kwargs: Dict) -> Dict[str, Any]:
        r"""Prepare the input, prompt_kwargs, model_kwargs for the model call."""
        # 1. render the prompt from the template
        prompt_str = self.get_prompt(**prompt_kwargs)
        # prompt = Prompt(template=self.template, prompt_kwargs=self.prompt_kwargs)

        # prompt_str = prompt.call(**prompt_kwargs).strip()

        # 2. combine the model_kwargs with the default model_kwargs
        composed_model_kwargs = self._compose_model_kwargs(**model_kwargs)

        max_tokens = composed_model_kwargs.get("max_tokens", None)
        prompt_tokens = self._tokenizer.count_tokens(prompt_str)
        self._estimated_token_count = prompt_tokens
        use_prompt_str = prompt_str
        if max_tokens is not None:
            if prompt_tokens > max_tokens:
                use_prompt_str = prompt_str[:max_tokens]
                log.warning(
                    f"Prompt is too long: {prompt_tokens} tokens, max tokens: {max_tokens}. Truncated prompt to: {use_prompt_str}"
                )
            # delete max_tokens from the model_kwargs
            del composed_model_kwargs["max_tokens"]

        # 3. convert app's inputs to api inputs
        api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
            # rename from input since input is a builtin object
            input=use_prompt_str,
            model_kwargs=composed_model_kwargs,
            model_type=self.model_type,
        )
        return api_kwargs

    def _model_client_call(self, api_kwargs: Dict, use_cache: bool = False) -> Any:
        # call the model client
        try:
            # check the cache
            index_content = json.dumps(api_kwargs)  # + f"training: {self.training}"
            if use_cache:
                # print(f"check cache first: {no_cache}")

                cached_completion = self._check_cache(index_content)
                if cached_completion is not None:
                    return cached_completion

            completion = self.model_client.call(
                api_kwargs=api_kwargs, model_type=self.model_type
            )

            # prepare cache - skip caching for streaming responses which contain unpickleable threading objects
            if use_cache and not api_kwargs.get("stream", False):
                self._save_cache(index_content, completion)
            return completion
        except Exception as e:
            log.error(f"Error calling the model: {e}")
            raise e

    async def _async_model_client_call(
        self, api_kwargs: Dict, use_cache: bool = False
    ) -> Any:
        # async call the model client with caching support
        try:
            # check the cache
            index_content = json.dumps(api_kwargs)  # + f"training: {self.training}"
            if use_cache:
                # Check cache first - cache operations are sync
                cached_completion = self._check_cache(index_content)
                if cached_completion is not None:
                    log.debug("Cache hit for async call")
                    return cached_completion

            completion = await self.model_client.acall(
                api_kwargs=api_kwargs, model_type=self.model_type
            )
            # save to cache - skip caching for streaming responses which contain unpickleable threading objects
            if use_cache and not api_kwargs.get("stream", False):
                self._save_cache(index_content, completion)
            return completion
        except Exception as e:
            log.error(f"Error calling the model: {e}")
            raise e

    ##############################################################################################################
    ### Forward, backwards, teacher generator, create demo data instance,
    # are for training and backpropagation
    ##############################################################################################################

    def create_demo_data_instance(
        self,
        input_prompt_kwargs: Dict[str, Any],
        output: GeneratorOutput,
        id: Optional[str] = None,
    ):
        r"""Automatically create a demo data instance from the input and output of the generator.
        Used to trace the demos for the demo paramter in the prompt_kwargs.
        Part of the few-shot learning.
        """
        from adalflow.core.base_data_class import DynamicDataClassFactory

        # map the input fields
        demo_data = {"id": id, "score": None}  # add score to trace the prediction score
        demo_data_class_output_mapping, output_fields = self._get_default_mapping(
            output
        )

        for k, v in input_prompt_kwargs.items():
            if isinstance(v, Parameter):
                demo_data[k] = v.map_to_successor(self)
            else:
                demo_data[k] = v
        # map the output fields
        for key, value in demo_data_class_output_mapping.items():
            demo_data[key] = value(output)

        obj = DynamicDataClassFactory.from_dict(demo_data)
        obj.set_input_fields([k for k in input_prompt_kwargs.keys()])
        obj.set_output_fields(output_fields)
        if obj is None:
            raise ValueError(f"Error creating the demo data instance:{demo_data}")
        return obj

    def set_backward_engine(self, backward_engine: "BackwardEngine" = None):
        if backward_engine is None:
            backward_engine = BackwardEngine(
                model_client=self.model_client,
                model_kwargs=self.model_kwargs,
            )
            if self.mock_output:
                backward_engine.set_mock_output()
        self.backward_engine = backward_engine

    def set_teacher_generator(self, teacher: "Generator" = None):
        self._teacher = teacher
        print(f"Teacher generator set: {self._teacher}, teacher {teacher}")
        log.debug(f"Teacher generator set: {self._teacher}")

    # def set_data_map_func(self, map_func: Callable = None):
    #     def default_map_func(data: "GeneratorOutputType") -> str:
    #         return (
    #             data.data
    #             if data.data
    #             else self.failure_message_to_backward_engine(data)
    #         )

    #     self.data_map_func = map_func or default_map_func

    #     log.debug(f"Data map function set: {self.data_map_func}")

    # TODO: limit to only one demo parameter.
    @staticmethod
    def find_demo_parameter(prompt_kwargs: Dict) -> Optional[Parameter]:
        from adalflow.optim.parameter import Parameter, ParameterType

        for p in prompt_kwargs.values():
            if isinstance(p, Parameter) and p.param_type == ParameterType.DEMOS:
                return p
        return None

    def forward(
        self,
        prompt_kwargs: Optional[
            Dict[str, Union[str, Parameter]]
        ] = {},  # the input need to be passed to the prompt
        model_kwargs: Optional[Dict] = {},
        id: Optional[str] = None,
    ) -> "Parameter":
        r"""Customized forward pass on top of the GradComponent forward method."""
        # 1. convert prompt_kwargs to parameter if it is not
        for k, v in prompt_kwargs.items():
            if not isinstance(v, Parameter):
                prompt_kwargs[k] = Parameter(
                    data=v,
                    name=f"{self.name}_{k}",
                    requires_opt=False,
                    param_type=ParameterType.INPUT,
                    data_id=id,
                )

        # 2. call the model
        unwrapped_prompt_kwargs: Dict[str, Any] = {}
        for k, v in prompt_kwargs.items():
            if isinstance(v, Parameter):
                if v.param_type == ParameterType.INPUT:
                    v.data_id = id
                unwrapped_prompt_kwargs[k] = v.map_to_successor(self)
            else:
                unwrapped_prompt_kwargs[k] = v
            log.debug(
                f"unwrapped_prompt_kwargs: {unwrapped_prompt_kwargs}, model_kwargs: {model_kwargs}"
            )
            log.debug(f"prompt template: {self.template}")

        output: GeneratorOutputType = None
        input_args = {}
        if self.mock_output:
            output = GeneratorOutput(data=self.mock_output_data)
        else:
            if self.teacher_mode and not isinstance(self, BackwardEngine):
                if not self._teacher:
                    log.debug(
                        f"unwrapped_prompt_kwargs: {unwrapped_prompt_kwargs}, model_kwargs: {model_kwargs}"
                    )
                    log.debug(f"names: {self.name}")
                    raise ValueError("Teacher generator is not set.")
                log.info(f"Using teacher: {self._teacher}")
                input_args = {
                    "prompt_kwargs": compose_model_kwargs(
                        self._teacher.prompt_kwargs, unwrapped_prompt_kwargs
                    ),
                    "model_kwargs": compose_model_kwargs(
                        self._teacher.model_kwargs, model_kwargs
                    ),
                }
                output = self._teacher.call(**input_args, id=id)
            else:
                input_args = {
                    "prompt_kwargs": compose_model_kwargs(
                        self.prompt_kwargs, unwrapped_prompt_kwargs
                    ),
                    "model_kwargs": compose_model_kwargs(
                        self.model_kwargs, model_kwargs
                    ),
                }

                output = self.call(**input_args, id=id)
                if not isinstance(output, GeneratorOutput):
                    raise ValueError(
                        f"Output should be of type GeneratorOutput, got {type(output)}"
                    )
        # 2. Generate a Parameter object from the output
        combined_prompt_kwargs = compose_model_kwargs(self.prompt_kwargs, prompt_kwargs)
        # if self.data_map_func is None:
        #     self.set_data_map_func()

        predecessors = [
            p for p in combined_prompt_kwargs.values() if isinstance(p, Parameter)
        ]

        log.debug(f"Predecessors: {predecessors} for generator {self.name}")

        def data_to_prompt_map_fn(data: Parameter) -> str:
            """GeneratorOutput will show the raw response instead of just the final data.
            The backward engine and optimizer should look at all reasoning to decide the gradient.
            """
            data: GeneratorOutput = data.data
            if data.error is not None:
                return f"Response: {data.raw_response} parsed with error: {data.error}"
            return f" {data.raw_response}"

        # TODO: all parameter should just wrap the whole output.
        # this is for training.
        param_data = output
        response: Parameter = OutputParameter(
            data=param_data,
            name=self.name + "_output",
            role_desc=f"Output from (llm) {self.name}",
            param_type=ParameterType.GENERATOR_OUTPUT,
            data_id=id,
            full_response=output,  # the data structure
            data_in_prompt=data_to_prompt_map_fn,
        )
        response.set_predecessors(predecessors)
        response.trace_forward_pass(
            input_args=input_args, full_response=output, id=self.id, name=self.name
        )
        # setattr(response, "full_response", output)
        # *** special to the generator ***
        response.trace_api_kwargs(api_kwargs=self._trace_api_kwargs)
        # attach the demo to the demo parameter
        # if self.tracing:
        demo_param = self.find_demo_parameter(combined_prompt_kwargs)

        if demo_param:
            if id is None:
                raise ValueError(
                    "ID is required for tracing. Please pass it to your Geneartor call."
                )

            demo = self.create_demo_data_instance(
                prompt_kwargs,
                output,
                id=id,
            )
            demo_param.add_dataclass_to_trace(demo, is_teacher=self.teacher_mode)
        else:
            log.debug(
                "No demo parameter found in the prompt_kwargs. You can not trace the demo data."
            )

        # **** end of the special to the generator ****

        # if not self.backward_engine:
        #     # self.set_backward_engine()
        #     log.debug(f"Backward engine: {self.backward_engine}")

        # attach a funtion to compute gradient for predecessors
        log.debug(f"disable_backward_engine: {self._disable_backward_engine}")

        response.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward,
                backward_engine=self.backward_engine,
                response=response,
                prompt_kwargs=prompt_kwargs,
                template=self.template,
                prompt_str=self.get_prompt(**combined_prompt_kwargs),
                disable_backward_engine=self._disable_backward_engine,
                id=id,
            )
        )
        return response

    def backward(
        self,
        response: Parameter,  # the output of the forward pass
        prompt_kwargs: Dict,
        template: str,
        prompt_str: str,
        backward_engine: Optional["Generator"] = None,
        id: Optional[str] = None,  # the id of the input
        disable_backward_engine: bool = False,
    ) -> Parameter:

        log.info(f"Generator: Backward: {response.name}")

        backward_pass_setup = (
            backward_engine.backward_pass_setup if backward_engine else None
        )
        log.debug(
            f"backward pass setup: {backward_pass_setup}, name: {self.name}",
            color="red",
        )

        children_params = response.predecessors
        is_intermediate_node = True
        if response.get_gradient_and_context_text().strip() == "":
            log.info(f"Generator: Backward: No gradient found for {response}.")

        # backward score to the demo parameter
        for pred in children_params:
            # if pred.requires_opt:
            if response.score is not None:
                pred.set_score(response.score)
            log.debug(
                f"backpropagate the score {response.score} to {pred.name}, is_teacher: {self.teacher_mode}"
            )
            if pred.param_type == ParameterType.DEMOS:
                # Accumulate the score to the demo
                pred.add_score_to_trace(
                    trace_id=id, score=response.score, is_teacher=self.teacher_mode
                )
                log.debug(f"Pred: {pred.name}, traces: {pred._traces}")

        # 1.backward for text-gradients
        if backward_engine:
            log.debug(
                f"Generator: Backward engine is set for the generator. {backward_engine}"
            )
            # if response.backward_engine_disabled:
            #     for pred in children_params:
            #         pred.backward_engine_disabled = True
            #     return

            all_pred_at_once = backward_pass_setup.all_pred_at_once

            if not all_pred_at_once:
                for pred in children_params:
                    if not pred.requires_opt or pred.param_type == ParameterType.DEMOS:
                        log.debug(
                            f"EvalFnToTextLoss: Skipping {pred} as it does not require optimization."
                        )
                        continue

                    self._backward_through_one_predecessor(
                        pred=pred,
                        response=response,
                        prompt_kwargs=prompt_kwargs,
                        # template=template,
                        backward_engine=backward_engine,
                        prompt_str=prompt_str,
                        backward_pass_setup=backward_pass_setup,
                        is_intermediate_node=is_intermediate_node,
                        disable_backward_engine=disable_backward_engine,
                    )
            else:
                backward = False
                for pred in children_params:
                    if pred.requires_opt and pred.param_type in [
                        ParameterType.PROMPT,
                        ParameterType.GENERATOR_OUTPUT,
                        ParameterType.RETRIEVER_OUTPUT,
                        ParameterType.OUTPUT,
                    ]:
                        backward = True
                        break
                if backward:
                    # 2nd approach, backward all that need opt at once.
                    self._backward_through_all_predecessors(
                        children_params=children_params,
                        response=response,
                        prompt_kwargs=prompt_kwargs,
                        template=template,
                        backward_engine=backward_engine,
                        prompt_str=prompt_str,
                        backward_pass_setup=backward_pass_setup,
                        is_intermediate_node=is_intermediate_node,
                    )
        else:
            log.debug("Backward engine is not set for the generator. No text gradient.")

    @staticmethod
    def _backward_through_all_predecessors(
        children_params: List[Parameter],
        response: Parameter,
        prompt_kwargs: Dict[str, str],
        backward_engine: "BackwardEngine",
        backward_pass_setup: BackwardPassSetup,
        is_intermediate_node: bool = False,
    ):
        parser = JsonParser()
        # instruction and objective is the same for all the children
        instruction_str, objective_str = None, None

        # 1. Generate the conversation input and output
        input_prompt_kwargs = {
            k: v.get_prompt_data() if isinstance(v, Parameter) else v
            for k, v in prompt_kwargs.items()
        }

        print(f"gt: {response.get_gt()}")

        # TODO: pass all the parameters and even the templates
        conversation_prompt_kwargs = {
            "input_value": input_prompt_kwargs,
            "llm_output": response.get_prompt_data(),
        }

        conversation_str = Prompt(
            prompt_kwargs=conversation_prompt_kwargs,
            template=LLM_CONVERSATION_TEMPLATE,
        )()

        all_pred_info = Prompt(
            prompt_kwargs={"variables": [p.get_param_info() for p in children_params]},
            template=ALL_PRED_INFO,
        )()

        conv_ins_template = None  # CONVERSATION_START_INSTRUCTION_BASE
        obj_ins_template = OBJECTIVE_INSTRUCTION_BASE
        if is_intermediate_node:  # TODO: this will always be true
            conv_ins_template = CONVERSATION_START_INSTRUCTION_CHAIN
            obj_ins_template = OBJECTIVE_INSTRUCTION_CHAIN
            response_gradient = response.get_gradients_str()
            # response_gradient = response.get_gradients_component_schema()
            # response_gradient = response.get_gradients_component_schema(
            #     skip_correct_sample=False
            # )
            if not response_gradient:
                raise ValueError(
                    f"Generator: No gradient found for {response}. Please check the response."
                )

        # replace variable and peers with all_pred_info

        instruction_str = Prompt(
            template=conv_ins_template,
            prompt_kwargs={
                "variable_and_peers_info": all_pred_info,
                "conversation_str": conversation_str,
            },
        )()
        objective_str = Prompt(
            template=obj_ins_template,
            prompt_kwargs={
                "response_desc": response.role_desc,
                "response_gradient": response_gradient,
                "instruction_to_backward_engine": response.instruction_to_backward_engine,
            },
        )()

        backward_engine_prompt_kwargs = {
            "conversation_sec": instruction_str,
            "objective_instruction_sec": objective_str,
            "output_format_str": OUTPUT_INSTRUCTION,
        }

        backward_engine_prompt_str = backward_engine.get_prompt(
            **backward_engine_prompt_kwargs
        )
        # print(f"Backward engine prompt: {backward_engine_prompt_str}")

        gradient_output: GeneratorOutput = None
        response_gradient_list = [""] * len(children_params)
        if (
            backward_pass_setup.compute_grad_for_errors_only
            and response.score is not None
            and float(response.score)
            > backward_pass_setup.threshold_score_to_compute_grad_for_errors
        ):
            manual_response_1 = f"Eval score: {response.score}. No noticeable error."
            response_gradient_list = [manual_response_1] * len(children_params)
            raw_response = str(response_gradient_list)
            gradient_output = GeneratorOutput(
                data=response_gradient_list, raw_response=raw_response
            )
        else:

            gradient_output: GeneratorOutput = backward_engine(
                prompt_kwargs=backward_engine_prompt_kwargs
            )
            if not isinstance(gradient_output, GeneratorOutput):
                raise ValueError(
                    f"Generator: Backward Engine should return a GeneratorOutput. Got {gradient_output} instead."
                )

            # parse the list of gradients

            try:
                response_gradient_list = parser.call(gradient_output.data)
            except Exception as e:
                log.error(f"Error parsing the response_gradient_list: {e}")
                failure_message = backward_engine.failure_message_to_optimizer(
                    gradient_output
                )
                if failure_message:
                    response_gradient_list = [failure_message] * len(children_params)

                log.debug(f"failure_message: {failure_message}")

        # computes gradient for each prompt predecessor
        for i, pred in enumerate(children_params):
            if not pred.requires_opt or pred.param_type == ParameterType.DEMOS:
                log.debug(
                    f"Generator: Skipping {pred} as it does not require optimization."
                )
                continue

            gradient_data = (
                response_gradient_list[i]
                if response_gradient_list and len(response_gradient_list) > i
                else "Failed to get the gradient."
            )

            var_gradient = Gradient(
                data=gradient_data,
                data_id=response.data_id,
                score=response.score,  # add score to gradient
                from_response=response,
                to_pred=pred,
            )
            var_gradient.add_context(
                GradientContext(
                    input_output=conversation_str,
                    response_desc=response.role_desc,
                    variable_desc=pred.role_desc,  # the only difference for each pred
                )
            )
            var_gradient.add_prompt(backward_engine_prompt_str)
            pred.add_gradient(var_gradient)
            if response.score is not None:
                pred.set_score(response.score)

    @staticmethod
    def _backward_through_one_predecessor(
        pred: Parameter,
        response: Parameter,
        prompt_kwargs: Dict[str, str],
        backward_engine: "BackwardEngine",
        prompt_str: str,
        backward_pass_setup: BackwardPassSetup,
        is_intermediate_node: bool = False,
        disable_backward_engine: bool = False,
    ):
        """Creating gradient/textual feedback for prompt type parameters."""
        if not pred.requires_opt:
            if response.score is not None:
                pred.set_score(response.score)
            log.debug(
                f"Generator: Skipping {pred} as it does not require optimization."
            )
            return

        if pred.check_if_already_computed_gradient_respect_to(response.id):
            log.debug(
                f"Generator: Skipping {pred} as the gradient is already computed."
            )

            return

        if backward_engine is None:
            log.error(
                "EvalFnToTextLoss: backward_engine is required for text prompt optimization."
            )
            raise ValueError(
                "EvalFnToTextLoss: backward_engine is required for text prompt optimization."
            )

        instruction_str, objective_str = None, None

        # 1. Generate the conversation string
        input_prompt_kwargs = {
            k: v.get_prompt_data() if isinstance(v, Parameter) else v
            for k, v in prompt_kwargs.items()
        }

        conversation_prompt_kwargs = {
            "input_value": input_prompt_kwargs,
            "llm_output": response.get_prompt_data(),
            "gt": response.get_gt(),
        }

        conversation_str = Prompt(
            prompt_kwargs=conversation_prompt_kwargs,
            template=LLM_CONVERSATION_TEMPLATE,
        )()

        variable_dict = pred.get_param_info()

        peers = [p.get_param_info() for p in pred.peers]

        variable_and_peers_info = Prompt(
            prompt_kwargs={"variable": variable_dict, "peers": peers},
            template=VARIABLE_AND_PEERS_INFO,
        )()

        # generator is almost always intermediate node
        conv_ins_template = None  # CONVERSATION_START_INSTRUCTION_BASE
        obj_ins_template = OBJECTIVE_INSTRUCTION_BASE
        if is_intermediate_node:  # TODO: this will always be true
            conv_ins_template = CONVERSATION_START_INSTRUCTION_CHAIN
            obj_ins_template = OBJECTIVE_INSTRUCTION_CHAIN
            response_gradient = response.get_gradients_str()
            # response_gradient = response.get_gradients_component_schema()
            if not response_gradient:
                raise ValueError(
                    f"Generator: No gradient found for {response}. Please check the response. pred: {pred}"
                )
        predecessors = [
            pred.get_param_info()
            for pred in response.predecessors
            if pred not in pred.peers
        ]
        instruction_str = Prompt(
            template=conv_ins_template,
            prompt_kwargs={
                "variable_and_peers_info": variable_and_peers_info,
                "conversation_str": conversation_str,
                "predecessors": predecessors,
            },
        )()
        log.info(f"Conversation start instruction base str: {instruction_str}")
        objective_str = Prompt(
            template=obj_ins_template,
            prompt_kwargs={
                "response_desc": response.role_desc,
                "response_gradient": response_gradient,
                "instruction_to_backward_engine": pred.instruction_to_backward_engine,
            },
        )()

        backward_engine_prompt_kwargs = {
            "conversation_sec": instruction_str,
            "objective_instruction_sec": objective_str,
        }
        backward_engine_prompt_str = backward_engine.get_prompt(
            **backward_engine_prompt_kwargs
        )
        # print(f"Backward engine prompt: {backward_engine_prompt_str}")
        gradient_value = None
        if not disable_backward_engine:
            gradient_output: GeneratorOutput = None
            if (
                backward_pass_setup.compute_grad_for_errors_only
                and response.score is not None
                and float(response.score)
                > backward_pass_setup.threshold_score_to_compute_grad_for_errors
            ):
                log.debug(
                    f"EvalFnToTextLoss: Skipping {pred} as the score is high enough."
                )
                # TODO: plus score descriptions
                manual_response = f"Eval score: {response.score}. No noticeable error."
                gradient_output = GeneratorOutput(
                    data=manual_response, raw_response=manual_response
                )
            else:

                gradient_output: GeneratorOutput = backward_engine(
                    prompt_kwargs=backward_engine_prompt_kwargs
                )
                prompt_str = backward_engine.get_prompt(  # noqa F841
                    **backward_engine_prompt_kwargs
                )
                # printc(f"Backward engine prompt: {prompt_str}")
                if not isinstance(gradient_output, GeneratorOutput):
                    raise ValueError(
                        f"Generator: Backward Engine should return a GeneratorOutput. Got {gradient_output} instead."
                    )
            # printc(f"Backward engine gradient: {gradient_output}")

            # USE this to trace each node's input and output, all nodes can be visualized
            log.info(
                f"Generator Backward Engine Prompt: {backward_engine.get_prompt( **backward_engine_prompt_kwargs)}"
            )
            gradient_value = (
                gradient_output.data
                or backward_engine.failure_message_to_optimizer(gradient_output)
            )
        var_gradient = Gradient(
            data=gradient_value,
            data_id=response.data_id,
            score=response.score,  # add score to gradient
            from_response=response,
            to_pred=pred,
        )
        # Component-level input and output.
        var_gradient.add_context(
            GradientContext(
                input_output=conversation_str,
                response_desc=response.role_desc,
                variable_desc=pred.role_desc,  # parameter_desc
            )
        )
        var_gradient.add_prompt(backward_engine_prompt_str)
        pred.add_gradient(var_gradient)
        if response.score is not None:
            pred.set_score(response.score)

    def _run_callbacks(
        self,
        output: GeneratorOutput,
        input: Dict,
        prompt_kwargs: Dict,
        model_kwargs: Dict,
    ):
        self.trigger_callbacks(
            "on_complete",
            output=output,
            input=input,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )
        if output.error:
            self.trigger_callbacks(
                "on_failure",
                output=output,
                input=input,
                prompt_kwargs=prompt_kwargs,
                model_kwargs=model_kwargs,
            )
        else:
            self.trigger_callbacks(
                "on_success",
                output=output,
                input=input,
                prompt_kwargs=prompt_kwargs,
                model_kwargs=model_kwargs,
            )

    def call(
        self,
        prompt_kwargs: Optional[Dict] = {},  # supports both str and parameter value
        model_kwargs: Optional[Dict] = {}, # can take images = {input: }, tools=[{"type": "image_generation"}], for image generation
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
    ) -> GeneratorOutputType:
        r"""
        Call the model_client by formatting prompt from the prompt_kwargs,
        and passing the combined model_kwargs to the model client.
        """
        prompt_str = self.get_prompt(**prompt_kwargs)
        with generator_span(
            generator_id="generator" + (id if id else ""),
            model_kwargs=self._compose_model_kwargs(**model_kwargs),
            prompt_kwargs=prompt_kwargs,
            prompt_template_with_keywords=prompt_str,
        ) as generator_span_data:
            generation_time = time.time()
            if self.mock_output:
                return GeneratorOutput(data=self.mock_output_data, id=id, input=prompt_str)

            log.debug(f"prompt_kwargs: {prompt_kwargs}")
            log.debug(f"model_kwargs: {model_kwargs}")

            api_kwargs = self._pre_call(prompt_kwargs, model_kwargs)

            log.debug(f"api_kwargs: {api_kwargs}")
            output: GeneratorOutputType = None
            # call the model client

            completion = None
            use_cache = use_cache if use_cache is not None else self._use_cache
            try:
                completion = self._model_client_call(
                    api_kwargs=api_kwargs, use_cache=use_cache
                )
            except Exception as e:
                log.error(f"Error calling the model: {e}")
                output = GeneratorOutput(error=str(e), id=id, input=prompt_str)
            # process the completion
            if completion is not None:
                try:
                    log.debug(f"Entering _post_call with completion: {completion}")
                    output = self._post_call(completion)
                except Exception as e:
                    log.error(f"Error processing the output: {e}")
                    output = GeneratorOutput(
                        raw_response=str(completion), error=str(e), id=id, input=prompt_str
                    )

            # User only need to use one of them, no need to use them all.
            output.id = id
            output.input = prompt_str
            self._run_callbacks(
                output,
                input=api_kwargs,
                prompt_kwargs=prompt_kwargs,
                model_kwargs=model_kwargs,
            )

            log.info(f"output: {output}")
            self._trace_api_kwargs = api_kwargs  # tracing

            generator_span_data.span_data.update_attributes(
                {"raw_response": output.raw_response}
            )
            generator_span_data.span_data.update_attributes(
                {"api_response": output.api_response}
            )
            generator_span_data.span_data.update_attributes(
                {"final_response": output.data}
            )
            generator_span_data.span_data.update_attributes(
                {"generation_time_in_seconds": time.time() - generation_time}
            )
            generator_span_data.span_data.update_attributes(
                {"token_usage": output.usage}
            )
            generator_span_data.span_data.update_attributes({"api_kwargs": api_kwargs})

            return output

    # TODO: training is not supported in async call yet
    async def acall(
        self,
        prompt_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
    ) -> Union[GeneratorOutputType, AsyncGenerator[GeneratorOutputType, None]]:
        r"""Async call the model with the input and model_kwargs.

        :warning::
            Training is not supported in async call yet.
        """
        prompt_str = self.get_prompt(**prompt_kwargs)

        with generator_span(
            generator_id="generator" + (id if id else ""),
            model_kwargs=self._compose_model_kwargs(**model_kwargs),
            prompt_kwargs=prompt_kwargs,
            prompt_template_with_keywords=prompt_str,
        ) as generator_span_data:

            if self.mock_output:
                generator_span_data.span_data.update_attributes(
                    {"final_response": self.mock_output_data}
                )
                return GeneratorOutput(data=self.mock_output_data, id=id, input=prompt_str)

            generation_time = time.time()
            log.info(f"prompt_kwargs: {prompt_kwargs}")
            log.info(f"model_kwargs: {model_kwargs}")

            api_kwargs = self._pre_call(prompt_kwargs, model_kwargs)
            output: GeneratorOutputType = None
            # call the model client
            completion = None
            use_cache = use_cache if use_cache is not None else self._use_cache

            log.info(f"api_kwargs: {api_kwargs}")

            try:
                completion = await self._async_model_client_call(
                    api_kwargs=api_kwargs, use_cache=use_cache
                )
            except Exception as e:
                log.error(f"Error calling the model: {e}")
                output = GeneratorOutput(error=str(e), id=id, input=prompt_str)

            if completion is not None:
                try:
                    # set ouput id in async post call instead
                    output = await self._async_post_call(completion)
                except Exception as e:
                    log.error(f"Error processing the output: {e}")
                    output = GeneratorOutput(
                        raw_response=str(completion), error=str(e), id=id, input=prompt_str
                    )

            # User only need to use one of them, no need to use them all.
            output.id = id
            output.input = prompt_str
            log.info(f"output: {output}")
            # if the output is not an async generator then set its id and run call backs
            # TODO support when output is an Async Generator
            if not isinstance(output, AsyncGenerator):
                self._run_callbacks(
                    output,
                    input=api_kwargs,
                    prompt_kwargs=prompt_kwargs,
                    model_kwargs=model_kwargs,
                )
                self._trace_api_kwargs = api_kwargs  # tracing

            # Update generator span attributes similar to call method
            generator_span_data.span_data.update_attributes(
                {"raw_response": output.raw_response if output else None}
            )
            generator_span_data.span_data.update_attributes(
                {"api_response": output.api_response if output else None}
            )
            generator_span_data.span_data.update_attributes(
                {"final_response": output.data if output else None}
            )
            generator_span_data.span_data.update_attributes(
                {"generation_time_in_seconds": time.time() - generation_time}
            )
            generator_span_data.span_data.update_attributes(
                {"token_usage": output.usage if output else None}
            )
            generator_span_data.span_data.update_attributes({"api_kwargs": api_kwargs})

            return output

    def __call__(self, *args, **kwargs) -> Union[GeneratorOutputType, Any]:
        if self.training:
            log.debug("Training mode")
            return self.forward(*args, **kwargs)
        else:
            log.debug("Inference mode")
            return self.call(*args, **kwargs)

    def _extra_repr(self) -> str:
        # Create the string for model_kwargs
        s = f"model_kwargs={self.model_kwargs}, "

        # Create the string for trainable prompt_kwargs
        prompt_kwargs_repr = [
            k
            for k, v in self.prompt_kwargs.items()
            if isinstance(v, Parameter) and v.requires_opt
        ]

        s += f"trainable_prompt_kwargs={prompt_kwargs_repr}"
        return s

    def to_dict(self) -> Dict[str, Any]:
        r"""Convert the generator to a dictionary."""
        # TODO: exclude default functions
        return super().to_dict()

    @staticmethod
    def failure_message_to_backward_engine(
        gradient_response: GeneratorOutput,
    ) -> Optional[str]:
        response_value = None
        if gradient_response.error or not gradient_response.data:
            response_value = f"Error: {gradient_response.error}, Raw response: {gradient_response.raw_response}"
        return response_value


class BackwardEngine(Generator):  # it is a generator with defaule template

    __doc__ = """A Generator with a default template for the backward pass in auto-differentiation.

    As a component, the forward pass is simply the same as the call method.
    So it will always return GeneratorOutputType instead of Parameter.

    If you want to customize the template, you can create your own backward engine.
    Yet, we will forever keep the training mode to False for the backward engine.
    This is achieved by making forward the same as call.
    """

    def __init__(self, **kwargs):
        if kwargs is None:
            kwargs = {}
        kwargs["template"] = FEEDBACK_ENGINE_TEMPLATE

        super().__init__(**kwargs)
        self.name = "BackwardEngine"
        self.teacher_mode = False

    def call(self, **kwargs) -> GeneratorOutputType:
        r"""Catch the rate limit error and raise it."""
        output = super().call(**kwargs)
        if output and output.error is not None and "429" in output.error:
            raise ValueError(f"Error in the backward engine: {output.error}")
        return output

    def forward(self, **kwargs):
        r"""Forward pass for the backward engine."""
        return self.call(**kwargs)

    @staticmethod
    def failure_message_to_optimizer(
        gradient_response: GeneratorOutput,
    ) -> Optional[str]:
        gradient_value_data = None
        if gradient_response.error or not gradient_response.data:
            gradient_value_data = f"The backward engine failed to compute the gradient. Raw response: {gradient_response.raw_response}, Error: {gradient_response.error}"

        return gradient_value_data


def create_teacher_generator(
    student: Generator,
    model_client: ModelClient,
    model_kwargs: Dict[str, Any],
    template: Optional[str] = None,
) -> Generator:
    r"""Create a teacher generator from the student generator.

    Note:
        Teacher generator will have no parameters.
        If you want to keep it to be the same as the student, just create one each time your student has been updated.
        Or else, task.parameters will list teacher parameters.

    Args:
        student (Generator): The student generator.
        model_client (ModelClient): The model client to use for the teacher generator.
        model_kwargs (Dict[str, Any]): The model kwargs to pass to the model client.
        name (str, optional): The name of the teacher generator. Defaults to "teacher".

    Returns:
        Generator: The teacher generator.
    """
    kwargs = student._kwargs.copy()
    kwargs["model_client"] = model_client
    kwargs["model_kwargs"] = model_kwargs
    if template:
        kwargs["template"] = template
    kwargs["name"] = f"{student.name}_teacher"

    prompt_kwargs_str: Dict[str, str] = {}
    for key, p in kwargs["prompt_kwargs"].items():
        if isinstance(p, Parameter):
            prompt_kwargs_str[key] = str(p.data)
        else:
            prompt_kwargs_str[key] = p
    kwargs["prompt_kwargs"] = prompt_kwargs_str
    teacher = Generator(
        **kwargs,
    )
    return teacher


if __name__ == "__main__":
    # test the generator with backward engine
    # TODO: move this to external local tests before packaging
    from adalflow.components.model_client import (
        GroqAPIClient,
        OpenAIClient,
        GoogleGenAIClient,
        AnthropicAPIClient,
    )
    from adalflow.utils import setup_env
    from adalflow.core.model_client import ModelClient

    setup_env()
    # log = get_logger(level="DEBUG")
    llama3_model = {
        "model_client": GroqAPIClient(),
        "model_kwargs": {
            "model": "llama-3.1-8b-instant",
        },
    }
    gpt_3_model = {
        "model_client": OpenAIClient(),
        "model_kwargs": {
            "model": "gpt-3.5-turbo",
        },
    }
    gemini_model = {
        "model_client": GoogleGenAIClient(),
        "model_kwargs": {
            "model": "gemini-1.0-pro",
        },
    }
    claude_model = {
        "model_client": AnthropicAPIClient(),
        "model_kwargs": {
            "model": "claude-3-opus-20240229",
            "max_tokens": 100,
        },
    }
    from adalflow.tracing.generator_call_logger import GeneratorCallLogger
    from functools import partial

    # setup the logger
    call_logger = GeneratorCallLogger(save_dir="traces")

    def on_complete(output, input, prompt_kwargs, model_kwargs, logger_call: Callable):
        logger_call(
            output=output,
            input=input,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

    for model in [llama3_model, gpt_3_model, gemini_model, claude_model]:
        generator = Generator(**model)

        teacher = create_teacher_generator(generator, **claude_model)

        call_logger.register_generator("generator", "generator_call")
        # setup the callback
        logger_call = partial(call_logger.log_call, name="generator")
        generator.register_callback(
            "on_complete", partial(on_complete, logger_call=logger_call)
        )

        output = generator(
            prompt_kwargs={
                "input_str": "Hello, world!",
            }
        )
        break

    # test the backward engine
    # TODO: test ollama and transformer client to update the change
