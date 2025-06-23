from adalflow.core.component import Component
from adalflow.core.generator import Generator
from adalflow.core.tool_manager import ToolManager
from adalflow.components.output_parsers.outputs import OutputParser
from adalflow.core.agent import Agent

from typing import Generator as GeneratorType, Dict, Optional, List, Any, Callable, Type, TypeVar, Generic, Union, Tuple
from dataclasses import dataclass, field
from adalflow.core.types import GeneratorOutput, FunctionOutput, StepOutput
from adalflow.optim.parameter import Parameter
from adalflow.core.types import Function
from pydantic import BaseModel
import logging

import asyncio

__all__ = ["Runner"]

log = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)  # Changed to use Pydantic BaseModel

@dataclass
class RunnerConfig:
    """Configuration for the Runner class.
    
    Attributes:
        output_type: Optional Pydantic data class type
        stream_parser: Optional stream parser
    """
    output_type: Optional[Type[T]] = None
    stream_parser: Optional["StreamParser"] = None

class Runner(Component):
    """A runner class that executes an adal.Agent instance with multi-step execution.
    
    Attributes:
        agent (Agent): The agent instance to execute
        config (RunnerConfig): Configuration for the runner
        max_steps (int): Maximum number of steps to execute
    """

    def __init__(
        self, 
        agent: Agent, 
        stream_parser: Optional["StreamParser"] = None,
        output_type: Optional[Type[T]] = None,
        max_steps: int = 10,
        **kwargs
    ) -> None:
        """Initialize runner with an agent and configuration.
        
        Args:
            agent: The agent instance to execute
            stream_parser: Optional stream parser
            output_type: Optional Pydantic data class type
            max_steps: Maximum number of steps to execute
        """
        super().__init__(**kwargs)
        self.agent = agent
        self.max_steps = max_steps
        self.config = RunnerConfig(
            stream_parser=stream_parser,
            output_type=output_type,
        )
        self.step_history = []
       

    def _process_function_calls(self, func: Function) -> FunctionOutput:
        """Process and execute function calls using the Agent's tool_manager.
        
        Args:
            func: The Function object containing the function call to execute
            
        Returns:
            str: The string representation of the function execution result
            
        Raises:
            ValueError: If the function name is not found in the tool_manager
            Exception: If function execution fails
        """
        try:
            # Get the tool from the tool manager
            tool = self.agent.tool_manager.get_tool(func.name)
            if tool is None:
                raise ValueError(f"Function '{func.name}' not found in tool_manager")
            
            # Execute the function with the provided arguments
            result = self.agent.tool_manager.execute_func(func)
            
            # Create FunctionOutput and convert to string
            func_output = FunctionOutput(
                name=func.name,
                input=func,
                output=result,
                error=None
            )
            
            # Convert to string format for prompt
            return func_output
            
        except Exception as e:
            log.error(f"Failed to execute function {func.name}: {str(e)}")
            return f"Function {func.name} failed to execute: {str(e)}"

    async def _aprocess_function_calls(self, func: Function) -> FunctionOutput:
        """Asynchronously process and execute function calls using the Agent's tool_manager.
        
        Args:
            func: The Function object containing the function call to execute
            
        Returns:
            FunctionOutput: The result of the function execution
        """
        try:
            # Get the tool from the tool manager
            tool = self.agent.tool_manager.get_tool(func.name)
            if tool is None:
                raise ValueError(f"Function '{func.name}' not found in tool_manager")
            
            # Execute the function directly on the current event loop
            result = await self.agent.tool_manager.aexecute_func(func)
            
            # Create FunctionOutput and return
            return FunctionOutput(
                name=func.name,
                input=func,
                output=result,
                error=None
            )
            
        except Exception as e:
            log.error(f"Failed to execute function {func.name}: {str(e)}", exc_info=True)
            return FunctionOutput(
                name=func.name,
                input=func,
                output=None,
                error=str(e)
            )

    def _check_last_step(
        self, step: StepOutput
    ) -> bool:
        """Check if the last step is the finish step.

        Args:
            step_history: List of previous steps

        Returns:
            bool: True if the last step is a finish step
        """
        
        assert(isinstance(step, StepOutput), f"Expected StepOutput, but got {type(step)}, value: {step}")
        
        # Check if it's a finish step
        if isinstance(step, StepOutput):
            action = step.action
            if action and (action.name == "finish" or getattr(action, "finish", False)):
                return True
        
        return False

    def _check_observation_step(
        self, step: StepOutput
    ) -> bool:
        """Check if the step is an observation step.

        Args:
            step_history: List of previous steps

        Returns:
            bool: True if the step is an observation step
        """
        
        assert(isinstance(step, StepOutput), f"Expected StepOutput, but got {type(step)}, value: {step}")
        
        # Check if it's an observation step
        if isinstance(step, StepOutput):
            action = step.action
            if action and (action.name == "observation" or getattr(action, "observation", False)):
                return True
        
        return False

    def _process_data(self, data: StepOutput, id: Optional[str] = None) -> T:
        """Process the generator output data field and convert to the specified pydantic data class of output_type.
        
        Args:
            data: The data to process
            id: Optional identifier for the output
            
        Returns:
            str: The processed data as a string
        """
        if not self.config.output_type:
            return data.observation

        try: 
            assert isinstance(data, StepOutput), f"Expected StepOutput, but got {type(data)}, value: {data}"
            assert isinstance(data.observation, dict), f"Expected data.observation to be a dictionary, but got {type(data.observation)}, value: {data.observation}" # expect data.observation to be a dictionary 
            # Convert to Pydantic model
            model_output = self.config.output_type(**data.observation)
            
            return model_output
            
        except Exception as e:
            log.error(f"Failed to parse output: {e}")
            return f"Error processing output: {str(e)}"

    def call(
        self,
        prompt_kwargs: Dict[str, Any],
        model_kwargs: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
    ) -> Tuple[List[GeneratorOutput], Any]:
        """Execute the agent synchronously for multiple steps with function calling support.
        
        Args:
            prompt_kwargs: Dictionary of prompt arguments for the generator
            model_kwargs: Optional model parameters to override defaults
            use_cache: Whether to use cached results if available
            id: Optional unique identifier for the request
            
        Returns:
            The generator output of type specified in self.config.output_type
        """
        self.step_history = []
        prompt_kwargs = prompt_kwargs.copy() if prompt_kwargs else {} 
        model_kwargs = model_kwargs.copy() if model_kwargs else {}
        step = 0
        last_output = None
        
        while step < self.max_steps:
            try:
                # Execute one step
                output = self.agent.call(
                    prompt_kwargs=prompt_kwargs,
                    model_kwargs=model_kwargs,
                    use_cache=use_cache,
                    id=id
                )

                if not isinstance(output, GeneratorOutput):
                    raise ValueError(
                        f"Expected GeneratorOutput, but got {type(output)}, value: {output}"
                    )
                
                self.step_history.append(output)
                
                # Process function calls if any
                if self.agent.is_training():
                    # agent generated response using generator's forward 
                    if not isinstance(output.data, GeneratorOutput):
                        raise ValueError(
                            f"Expected GeneratorOutput for output.data, but got {type(output.data)}, value: {output.data}"
                        )
                    step = output.data.data 
                else: 
                    # agent generated response using generator's call (inference)
                    step = output.data 

                assert(isinstance(step, StepOutput), f"Expected OutputProcessor to return StepOutput type, but got {type(step)}, value: {step}")

                if self._check_last_step(step):
                    return self.step_history, self._process_data(step)

                # Check if the action is a Function and its name is 'finish'
                if step.function and isinstance(step.function, Function): 

                    function_results = self._process_function_calls(step.function)
                    # Add function results to prompt for next step
                    prompt_kwargs['function_results'] = str(function_results) # use pydantic data class's __str__ method 
                    continue
                elif self._check_observation_step(step):
                    # Process the output
                    processed = self._process_data(step, id) 
                    last_output = processed
                    # wrap previous output in prompt_kwargs and add prompt data from previous output 
                    # use pydantic data class's __str__ method 
                    prompt_kwargs['previous_output'] = str(processed) if isinstance(processed, self.config.output_type) else processed
                    
                step += 1
                    
            except Exception as e:
                log.error(f"Error in step {step}: {str(e)}")
                return f"Error in step {step}: {str(e)}"

        return self.step_history, last_output

    async def acall(
        self,
        prompt_kwargs: Dict[str, Any],
        model_kwargs: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
    ) -> Tuple[List[GeneratorOutput], T]:
        """Execute the agent asynchronously for multiple steps with function calling support.
        
        Args:
            prompt_kwargs: Dictionary of prompt arguments for the generator
            model_kwargs: Optional model parameters to override defaults
            use_cache: Whether to use cached results if available
            id: Optional unique identifier for the request
            
        Returns:
            Tuple containing:
                - List of step history (GeneratorOutput objects)
                - Final processed output
        """
        self.step_history = []
        prompt_kwargs = prompt_kwargs.copy() if prompt_kwargs else {}
        model_kwargs = model_kwargs.copy() if model_kwargs else {}
        step_count = 0
        last_output = None
        
        while step_count < self.max_steps:
            try:
                # Execute one step asynchronously
                output = await self.agent.acall(
                    prompt_kwargs=prompt_kwargs,
                    model_kwargs=model_kwargs,
                    use_cache=use_cache,
                    id=id
                )

                if not isinstance(output, GeneratorOutput):
                    raise ValueError(
                        f"Expected GeneratorOutput, but got {type(output)}, value: {output}"
                    )
                
                self.step_history.append(output)
                
                # Process function calls if any
                if self.agent.is_training():
                    # agent generated response using generator's forward 
                    if not isinstance(output.data, GeneratorOutput):
                        raise ValueError(
                            f"Expected GeneratorOutput for output.data, but got {type(output.data)}, value: {output.data}"
                        )
                    step = output.data.data 
                else: 
                    # agent generated response using generator's call (inference)
                    step = output.data 

                assert(isinstance(step, StepOutput), 
                    f"Expected OutputProcessor to return StepOutput type, but got {type(step)}, value: {step}")

                # Check if the action is a Function and its name is 'finish'
                if step.function and isinstance(step.function, Function): 
                    if self._check_last_step(step):
                        return self.step_history, self._process_data(step)

                    function_results = await self._aprocess_function_calls(step.function)
                    # Add function results to prompt for next step
                    prompt_kwargs['function_results'] = str(function_results)
                    continue
                elif self._check_observation_step(step):
                    # Process the output
                    processed = self._process_data(step, id) 
                    last_output = processed
                    # wrap previous output in prompt_kwargs and add prompt data from previous output 
                    prompt_kwargs['previous_output'] = str(processed) if isinstance(processed, self.config.output_type) else processed
                    
                step_count += 1
                    
            except Exception as e:
                log.error(f"Error in step {step_count}: {str(e)}")
                return f"Error in step {step_count}: {str(e)}"

        return self.step_history, last_output

    def stream(
        self,
        user_query: str, 
        current_objective: Optional[str] = None,
        memory: Optional[str] = None,
    ) -> GeneratorType[Any, None, None]:
        """
        Synchronously executes the agent output and stream results.
        Optionally parse and post-process each chunk.
        """
        # TODO replace Any type with StreamChunk type 
        try: 
            generator_output = self.call(user_query, current_objective, memory)

            if self.config.stream_parser:
                for chunk in generator_output.data:
                    yield self.config.stream_parser(chunk)
            else: 
                log.error("StreamParser not specified")
                for chunk in generator_output.data:
                    yield chunk 
                # TODO: need to define a StreamChunk type in library

        except Exception as e: 
            log.error(f"Failed to stream generator output: {e}")
            yield generator_output
            # TODO: need to define a StreamChunk type in library
            

        
    # TODO implement async stream 
    async def astream( 
        self,
        user_query: str, 
        current_objective: Optional[str] = None,
        memory: Optional[str] = None,
    ) -> GeneratorType[Any, None, None]:
        """
        Execute the agent asynchronously and stream results.
        Optionally parse and post-process each chunk.
        """
        # TODO replace Any type with StreamChunk type 
        ... 
        # This would require relying on the async_stream of the model_client instance of the generator and parsing that 
        # using custom logic to buffer chunks and only stream when they complete a certain top-level field

    def backward(
        self,
        response: Parameter,
        prompt_kwargs: Optional[Dict] = None,
        template: Optional[str] = None,
        backward_engine: Optional["Generator"] = None,
        id: Optional[str] = None,
        disable_backward_engine: bool = False
    ):
        """
        Run backward pass on the agent.
        Template is expected to be the template to guide the backward pass. 
        """
        return self.agent.backward(
            response=response,
            prompt_kwargs=prompt_kwargs,
            template=template,
            backward_engine=backward_engine,
            id=id,
            disable_backward_engine=disable_backward_engine
        )

    def update_runner(
        self,
        agent: Optional[Agent] = None,
        stream_parser: Optional["StreamParser"] = None,
        output_parser: Optional[OutputParser] = None,
        output_class: Optional[Type] = None,
        context_map: Optional[Dict[str, Function]] = None,
        agent_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update runner configuration where the user can optionally provide a new agent instance or 
        a configuration of the agent if it is to be updated."""

        # if both agent instance and agent_config is provided update using agent 
        if agent is not None: 
            self.agent = agent
        else: 
            if agent_config is not None:
                self.agent = Agent.from_config(agent_config)

        # keep as is if None 
        self.config = RunnerConfig(
            stream_parser=stream_parser or self.config.stream_parser,
            output_parser=output_parser or self.config.output_parser,
            output_class=output_class or self.config.output_class,
            context_map=context_map or self.config.context_map,
        )
        
    """ 
	  internal tool to execute tools as necessary based on the generator output of call and stream_call
    """ 
    def _tool_execute(
        self,
        func: Function,
    ) -> Union[FunctionOutput, Parameter]:
        """
        Execute a tool function through the agent's tool manager.
        Handles both sync and async functions.
        """
        return self.agent.tool_manager.call(expr_or_fun=func, step="execute")

        # except Exception as e: 
        #     # TODO check map_f 
        #     if func is not None and isinstance(func, Function) and map_fn is None:
        #         function_call_response = asyncio.run(
        #             self.agent.context_map[func.name](**func.kwargs)
        #         )
        #         return function_call_response
        #     else: 
        #         raise ValueError(f"Error {e} executing function: {func}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Runner':
        """Create a Runner instance from a configuration dictionary.
        
        Args:
            config: Configuration dictionary containing:
                - agent: Agent configuration 
                - stream_parser: Optional stream parser
                - output_parser: Optional output parser
                - output_class: Optional output class
                - context_map: Optional context map
                - name: Optional name for the runner

        Example 1: Create a Runner from a config dictionary
        runner_config = {
            'name': 'example_runner',
            'agent': {
                'name': 'example_agent',
                'system_prompt': 'You are a helpful assistant.',
                'model_client': {
                    'component_name': 'OpenAIClient',
                    'component_config': {
                'api_key': 'your-api-key',
                        'model': 'gpt-3.5-turbo'
                    }
                },
                'tool_manager': {
                    'tools': []  # List of tools would go here
                }
            },
            'output_parser': lambda x: {'text': x.text},  # Simple output parser
            'output_class': GeneratorOutput,
        }
                
        Returns:
            Configured Runner instance
        """
        
        # Extract agent config/instance
        try: 
            agent = Agent.from_config(config.get('agent'))
        except Exception as e: 
            raise ValueError(f"Failed to create agent from config: {e}")
        
        # Create runner instance
        runner = cls(
            agent=agent,
            stream_parser=config.get('stream_parser', None),
            output_parser=config.get('output_parser', None),
            output_class=config.get('output_class', GeneratorOutput),
        )
        
        return runner

    def return_state_dict(self) -> Dict[str, Any]:
        """Return the state of the runner as a dictionary that can be used to recreate it.
        
        Returns:
            Dictionary containing the runner's state
        """
        return {
            'agent': self.agent.return_state_dict(),
            'stream_parser': self.config.stream_parser,
            'output_parser': self.config.output_parser,
            'output_class': self.config.output_class,
            'class_name': self.__class__.__name__,
            'module_name': self.__class__.__module__
        }