"""Generator is a user-facing orchestration component with a simple and unified interface for LLM prediction.

It is a pipeline that consists of three subcomponents."""

import os
import json

from typing import Any, Dict, Optional, Union, Callable
from copy import deepcopy
import logging


from lightrag.core.types import (
    ModelType,
    GeneratorOutput,
    GeneratorOutputType,
)
from lightrag.core.component import Component
from lightrag.core.base_data_class import DataClass, check_adal_dataclass


from lightrag.optim.parameter import Parameter, GradientContext
from lightrag.optim.types import ParameterType

from lightrag.core.prompt_builder import Prompt
from lightrag.core.functional import compose_model_kwargs
from lightrag.core.model_client import ModelClient
from lightrag.core.default_prompt_template import DEFAULT_LIGHTRAG_SYSTEM_PROMPT
from lightrag.optim.function import BackwardContext, GradFunction
from lightrag.utils.cache import CachedEngine
from lightrag.tracing.callback_manager import CallbackManager
from lightrag.utils.global_config import get_adalflow_default_root_path

from lightrag.optim.text_grad.backend_engine_prompt import (
    FEEDBACK_ENGINE_TEMPLATE,
    BACKWARD_SYSTEM_PROMPT,
    CONVERSATION_TEMPLATE,
    CONVERSATION_START_INSTRUCTION_BASE,
    CONVERSATION_START_INSTRUCTION_CHAIN,
    # EVALUATE_VARIABLE_INSTRUCTION,
    OBJECTIVE_INSTRUCTION_BASE,
    OBJECTIVE_INSTRUCTION_CHAIN,
)

log = logging.getLogger(__name__)


class Generator(GradFunction, CachedEngine, CallbackManager):
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

    _use_cache: bool = False
    _kwargs: Dict[str, Any] = {}

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
        cache_path: Optional[str] = None,
        use_cache: bool = False,
        demo_data_class: Optional[DataClass] = None,
        demo_data_class_input_mapping: Optional[
            Dict[str, str]
        ] = {},  # prompt_kwargs key to demo_data_class field
        demo_data_class_output_mapping: Optional[
            Dict[str, Callable]
        ] = {},  # GeneratorOut will be matched to demo_data_class field via a Callable
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

        # Cache
        model_str = (
            f"{model_client.__class__.__name__}_{model_kwargs.get('model', 'default')}"
        )
        _cache_path = (
            get_adalflow_default_root_path() if cache_path is None else cache_path
        )
        self.cache_path = os.path.join(_cache_path, f"cache_{model_str}.db")

        print(f"cache_path: {self.cache_path}")

        CachedEngine.__init__(self, cache_path=self.cache_path)
        Component.__init__(self)
        GradFunction.__init__(self)
        CallbackManager.__init__(self)

        self.name = name or self.__class__.__name__

        self._init_prompt(template, prompt_kwargs)

        self.model_kwargs = model_kwargs.copy()
        # init the model client
        self.model_client = model_client

        self.output_processors = output_processors

        # add trainable_params to generator
        for key, p in prompt_kwargs.items():
            if isinstance(p, Parameter):
                # peers will be all other parameters
                peers = [
                    p
                    for k, p in prompt_kwargs.items()
                    if isinstance(p, Parameter) and k != key
                ]
                p.set_peers(peers)
                setattr(self, key, p)

        # end of trainable parameters
        self.backward_engine: "BackwardEngine" = None
        log.info(f"Generator {self.name} initialized.")
        #  to support better testing on the parts beside of the model call
        self.mock_output: bool = False
        self.mock_output_data: str = "mock data"
        self.data_map_func: Callable = None
        self.set_data_map_func()
        self.model_str = model_str
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
        if demo_data_class:
            check_adal_dataclass(demo_data_class)
        self._demo_data_class = demo_data_class
        self._demo_data_class_input_mapping = demo_data_class_input_mapping
        self._demo_data_class_output_mapping = demo_data_class_output_mapping

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
        # NOTE: Prompt can handle parameters
        self.prompt = Prompt(template=template, prompt_kwargs=self.prompt_kwargs)

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
        # prompt_kwargs_str = _convert_prompt_kwargs_to_str(kwargs)
        return self.prompt.print_prompt(**kwargs)

    def get_prompt(self, **kwargs) -> str:
        # prompt_kwargs_str = _convert_prompt_kwargs_to_str(kwargs)
        return self.prompt.call(**kwargs)

    def _extra_repr(self) -> str:
        s = f"model_kwargs={self.model_kwargs}, model_type={self.model_type}"
        return s

    def _post_call(self, completion: Any) -> GeneratorOutput:
        r"""Get string completion and process it with the output_processors."""
        output: GeneratorOutput = self.model_client.parse_chat_completion(completion)
        # the output processors operate on the response, most cases it is a string.
        data = output.data
        if self.output_processors and data:
            try:
                data = self.output_processors(data)
                output.data = data
            except Exception as e:
                log.error(f"Error processing the output processors: {e}")
                output.error = str(e)

        return output

    def _pre_call(self, prompt_kwargs: Dict, model_kwargs: Dict) -> Dict[str, Any]:
        r"""Prepare the input, prompt_kwargs, model_kwargs for the model call."""
        # 1. render the prompt from the template
        prompt_str = self.prompt.call(**prompt_kwargs).strip()

        # 2. combine the model_kwargs with the default model_kwargs
        composed_model_kwargs = self._compose_model_kwargs(**model_kwargs)

        # 3. convert app's inputs to api inputs
        api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
            input=prompt_str,
            model_kwargs=composed_model_kwargs,
            model_type=self.model_type,
        )
        return api_kwargs

    def _model_client_call(self, api_kwargs: Dict, use_cache: bool = False) -> Any:
        # call the model client
        try:
            # check the cache
            index_content = json.dumps(api_kwargs)  # all messages
            if use_cache:
                # print(f"check cache first: {no_cache}")

                cached_completion = self._check_cache(index_content)
                if cached_completion is not None:
                    return cached_completion

            completion = self.model_client.call(
                api_kwargs=api_kwargs, model_type=self.model_type
            )
            # prepare cache
            if use_cache:
                self._save_cache(index_content, completion)
            return completion
        except Exception as e:
            log.error(f"Error calling the model: {e}")
            raise e

    ##############################################################################################################
    ### Forward and backwards, and teacher generator are for training
    ##############################################################################################################
    @staticmethod
    def create_demo_data_instance(
        demo_class: DataClass,
        prompt_kwargs: Dict[str, Any],
        output: GeneratorOutput,
        demo_data_class_input_mapping: Dict[str, str],
        demo_data_class_output_mapping: Dict[str, Any],
        id: Optional[str] = None,
    ):
        check_adal_dataclass(demo_class)
        # map the input fields
        demo_data = {"id": id}
        for key, value in demo_data_class_input_mapping.items():
            demo_data[key] = prompt_kwargs[value]
        # map the output fields
        for key, value in demo_data_class_output_mapping.items():
            demo_data[key] = value(output)
        obj = demo_class.from_dict(demo_data)
        if obj is None:
            raise ValueError(
                f"Error creating the demo data instance: {demo_class}, {demo_data}"
            )
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

    def set_data_map_func(self, map_func: Callable = None):
        def default_map_func(data: "GeneratorOutputType") -> str:
            return (
                data.data
                if data.data
                else self.failure_message_to_backward_engine(data)
            )

        self.data_map_func = map_func or default_map_func

        log.debug(f"Data map function set: {self.data_map_func}")

    # TODO: limit to only one demo parameter.
    @staticmethod
    def find_demo_parameter(prompt_kwargs: Dict) -> Optional[Parameter]:
        from lightrag.optim.parameter import Parameter, ParameterType

        for p in prompt_kwargs.values():
            if isinstance(p, Parameter) and p.param_type == ParameterType.DEMOS:
                return p
        return None

    # NOTE: when training is true, forward will be called in __call__ instead of call
    def forward(
        self,
        prompt_kwargs: Optional[Dict] = {},  # the input need to be passed to the prompt
        model_kwargs: Optional[Dict] = {},
        id: Optional[str] = None,
    ) -> "Parameter":
        # 1. call the model
        output: GeneratorOutputType = None
        input_args = {}
        if self.mock_output:
            output = GeneratorOutput(data=self.mock_output_data)
        else:
            if self.teacher_mode:
                if not self._teacher:
                    raise ValueError("Teacher generator is not set.")
                log.info(f"Using teacher: {self._teacher}")
                input_args = {
                    "prompt_kwargs": compose_model_kwargs(
                        self._teacher.prompt_kwargs, prompt_kwargs
                    ),
                    "model_kwargs": compose_model_kwargs(
                        self._teacher.model_kwargs, model_kwargs
                    ),
                }
                output = self._teacher.call(prompt_kwargs, model_kwargs)
            else:
                input_args = {
                    "prompt_kwargs": compose_model_kwargs(
                        self.prompt_kwargs, prompt_kwargs
                    ),
                    "model_kwargs": compose_model_kwargs(
                        self.model_kwargs, model_kwargs
                    ),
                }
                output = self.call(prompt_kwargs, model_kwargs)
        # 2. Generate a Parameter object from the output
        combined_prompt_kwargs = compose_model_kwargs(self.prompt_kwargs, prompt_kwargs)
        if self.data_map_func is None:
            self.set_data_map_func()

        predecessors = [
            p for p in combined_prompt_kwargs.values() if isinstance(p, Parameter)
        ]

        log.debug(f"Predecessors: {predecessors} for generator {self.name}")
        response: Parameter = Parameter(
            data=self.data_map_func(output),
            alias=self.name + "_output",
            predecessors=predecessors,
            role_desc=f"response from generator {self.name}",
            # context of the forward pass
            input_args=input_args,
            raw_response=output.raw_response,
        )
        # attach the demo to the demo parameter
        if self.tracing:
            demo_param = self.find_demo_parameter(combined_prompt_kwargs)
            if id is None:
                raise ValueError(
                    "ID is required for tracing. Please pass it to your Geneartor call."
                )
            if demo_param:

                demo = self.create_demo_data_instance(
                    self._demo_data_class,
                    combined_prompt_kwargs,
                    output,
                    self._demo_data_class_input_mapping,
                    self._demo_data_class_output_mapping,
                    id=id,
                )
                demo_param.add_to_trace(demo, is_teacher=self.teacher_mode)
            else:
                log.warning(
                    "No demo parameter found in the prompt_kwargs. You can not trace the demo data."
                )
                # raise ValueError(
                #     "No demo parameter found in the prompt_kwargs. You can not trace the demo data."
                # )
        if not self.backward_engine:
            # self.set_backward_engine()
            log.debug(f"Backward engine: {self.backward_engine}")

        # attach a funtion to compute gradient for predecessors
        response.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward,
                backward_engine=self.backward_engine,
                response=response,
                prompt_kwargs=combined_prompt_kwargs,
                prompt_str=self.get_prompt(**combined_prompt_kwargs),
                id=id,
            )
        )
        # attach a function to backpropagate the evaluation score to precessor demos.
        return response

    # == pytorch custom autograd function ==
    def backward(
        self,
        response: Parameter,  # the output of the forward pass
        prompt_kwargs: Dict,
        prompt_str: str,
        backward_engine: Optional["Generator"] = None,
        id: Optional[str] = None,  # the id of the input
    ) -> Parameter:

        log.info(f"Generator: Backward: {response}")

        children_params = response.predecessors
        is_chain = True
        if response.get_gradient_and_context_text().strip() == "":
            log.info(f"Generator: Backward: No gradient found for {response}.")
        # Compute all predecessors's gradients based on the current response' note.

        # 1.backward for text-gradients
        if backward_engine:
            log.debug(
                f"Generator: Backward engine is set for the generator. {backward_engine}"
            )
            for pred in children_params:
                # NOTE: not requires_opt should only be used to skip the optimization, not the backpropagation
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
        else:
            log.debug("Backward engine is not set for the generator. No text gradient.")
        # backward score to the demo parameter
        for pred in children_params:
            if pred.requires_opt:
                # pred._score = float(response._score)
                pred.set_score(response._score)
                log.debug(
                    f"backpropagate the score {response._score} to {pred.alias}, is_teacher: {self.teacher_mode}"
                )
                if pred.param_type == ParameterType.DEMOS:
                    # Accumulate the score to the demo
                    pred.add_score_to_trace(
                        trace_id=id, score=response._score, is_teacher=self.teacher_mode
                    )
                    log.debug(f"Pred: {pred.alias}, traces: {pred._traces}")

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
                "variable_value": pred.raw_response or pred.data,
                "param_type": pred.param_type,
                "instruction_to_backward_engine": pred.instruction_to_backward_engine,
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
        # evaluation_variable_instruction_str = Prompt(
        #     template=EVALUATE_VARIABLE_INSTRUCTION,
        #     prompt_kwargs={
        #         "variable_desc": pred.role_desc,
        #         "variable_short": pred.raw_response or pred.data,
        #     },
        # )()

        # log.info(
        #     f"Evaluation variable instruction str: {evaluation_variable_instruction_str}"
        # )
        backward_engine_prompt_kwargs = {
            "BACKWARD_SYSTEM_PROMPT": Prompt(BACKWARD_SYSTEM_PROMPT)(),
            "conversation_sec": instruction_str,
            "objective_instruction_sec": objective_str,
            # "evaluate_variable_instruction_sec": evaluation_variable_instruction_str,
        }

        gradient_output: GeneratorOutput = backward_engine(
            prompt_kwargs=backward_engine_prompt_kwargs
        )
        # USE this to trace each node's input and output, all nodes can be visualized
        log.info(
            f"Generator Backward Engine Prompt: {backward_engine.get_prompt( **backward_engine_prompt_kwargs)}"
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
        prompt_str = backward_engine.get_prompt(**backward_engine_prompt_kwargs)

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

    def _run_callbacks(self, output: GeneratorOutput, input: Dict):
        self.trigger_callbacks(
            "on_complete",
            output=output,
            input=input,
        )
        if output.error:
            self.trigger_callbacks(
                "on_failure",
                output=output,
                input=input,
            )
        else:
            self.trigger_callbacks(
                "on_success",
                output=output,
                input=input,
            )

    def call(
        self,
        prompt_kwargs: Optional[Dict] = {},  # the input need to be passed to the prompt
        model_kwargs: Optional[Dict] = {},
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
    ) -> GeneratorOutputType:
        r"""
        Call the model_client by formatting prompt from the prompt_kwargs,
        and passing the combined model_kwargs to the model client.
        """
        if self.mock_output:
            return GeneratorOutput(data=self.mock_output_data, id=id)

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
            output = GeneratorOutput(error=str(e), id=id)
        # process the completion
        if completion is not None:
            try:
                output = self._post_call(completion)

            except Exception as e:
                log.error(f"Error processing the output: {e}")
                output = GeneratorOutput(
                    raw_response=str(completion), error=str(e), id=id
                )

        # User only need to use one of them, no need to use them all.
        self._run_callbacks(output, input=api_kwargs)
        output.id = id

        log.info(f"output: {output}")
        return output

    # TODO: training is not supported in async call yet
    async def acall(
        self,
        prompt_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
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
        output.id = id
        log.info(f"output: {output}")
        self._run_callbacks(output, input=api_kwargs)
        return output

    def __call__(self, *args, **kwargs) -> Union[GeneratorOutputType, Any]:
        if self.training:
            log.debug("Training mode")
            print("Training mode")
            return self.forward(*args, **kwargs)
        else:
            log.debug("Inference mode")
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
    from lightrag.components.model_client import (
        GroqAPIClient,
        OpenAIClient,
        GoogleGenAIClient,
        AnthropicAPIClient,
    )
    from lightrag.utils import setup_env
    from lightrag.core.model_client import ModelClient

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
    from lightrag.tracing.generator_call_logger import GeneratorCallLogger
    from functools import partial

    # setup the logger
    call_logger = GeneratorCallLogger(save_dir="traces")

    def on_complete(output, input, prompt_kwargs, model_kwargs, logger_call: Callable):
        print(f"on_complet  output: {output}")
        logger_call(
            output=output,
            input=input,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

    for model in [llama3_model, gpt_3_model, gemini_model, claude_model]:
        print(f"""model: {model["model_kwargs"]["model"]}""")
        generator = Generator(**model)

        print("_kwargs: ", generator._kwargs)

        teacher = create_teacher_generator(generator, **claude_model)
        print(f"teacher: {teacher}")

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
        print(f"output: {output}")
        break

    # test the backward engine
    # TODO: test ollama and transformer client to update the change
