from adal import Component
from adalflow import Generator
from adalflow.core.tool_manager import ToolManager
from adalflow.core.output_parser import OutputParser
from adalflow.core.agent import Agent

from typing import Dict, Optional, List, Any, Callable, Type, TypeVar, Generic
from dataclasses import dataclass, field
from adalflow.core.types import GeneratorOutput
import logging

import asyncio

__all__ = ["Runner"]

T = TypeVar('T', bound=GeneratorOutput)


log = logging.getLogger(__name__)

@dataclass
class RunnerConfig:
    """Configuration for the Runner class.
    
    Attributes:
        output_parser: Optional dictionary of parse functions that parse the necessary attributes to the parsed class
        output_class: Optional output class type
        context_map: Optional context map
        stream_parser: Optional stream parser
    """
    output_parser: Optional[Callable[[GeneratorOutput], Dict[str, Any]]] = None
    output_class: Optional[Type[T]] = GeneratorOutput
    stream_parser: Optional["StreamParser"] = None

class Runner(Component):
    """A runner class that executes an adal.Agent instance.
    
    Attributes:
        agent (Agent): The agent instance to execute
        config (RunnerConfig): Configuration for the runner
    """

    def __init__(
        self, 
        agent: Agent, 
        stream_parser: Optional["StreamParser"] = None,
        output_parser: Optional[Callable[[GeneratorOutput], Dict[str, Any]]] = None,
        output_class: Optional[Type[T]] = GeneratorOutput,
        **kwargs
    ) -> None:
        """Initialize runner with an agent and configuration.
        
        Args:
            agent: The agent instance to execute
            stream_parser: Optional stream parser
            output_parser: Optional dictionary of parse functions that parse the necessary attributes to the parsed class
            output_class: Optional output class type
            context_map: Optional context map
            stream_parser: Optional stream parser
        """
        self.agent = agent
        self.config = RunnerConfig(
            stream_parser=stream_parser,
            output_parser=output_parser,
            output_class=output_class, 
        )
        super().__init__(**kwargs)

    def call(
        self,
        user_query: str,
        current_objective: Optional[str] = None,
        memory: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
    ) -> T:
        """Execute the agent synchronously and return generator output after parsing to desire output class type 
        
        Args:
            user_query: The user's input query
            current_objective: Optional current objective/context
            memory: Optional memory/chat history
            model_kwargs: Optional model parameters to override defaults
            use_cache: Whether to use cached results if available
            id: Optional unique identifier for the request
            
        Returns:
            The generator output of type specified in self.config.output_class
        """
        generator_output = self.agent(
            user_query=user_query,
            current_objective=current_objective,
            memory=memory,
            model_kwargs=model_kwargs,
            use_cache=use_cache,
            id=id
        )

        if not self.config.output_parser: 
            return generator_output

        # parse function is a generic utility function that takes a generator output and returns a dictionary of attributes 
        try: 
            parsed_output = self.config.output_class()
            attrs = self.config.output_parser(generator_output)
            # set attributes on the parsed output object to the class 
            for attr, value in attrs.items():
                setattr(parsed_output, attr, value)

            # TODO optionally execute tools based on the parsed attributes 

        except Exception as e:
            log.error(f"Failed to parse and apply post process functions on generator output: {e}")
            # default to returning an object of the generator type with error e
            return GeneratorOutput(error=str(e), id=id)

        return parsed_output 

    def acall (
        self,
        user_query: str,
        current_objective: Optional[str] = None,
        memory: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = {},
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
    ) -> GeneratorOutput:
        """Call the generator with the given arguments."""
        generator_output = self.agent.acall(user_query, current_objective, memory, model_kwargs, use_cache, id)

        if not self.config.output_parser: 
            return generator_output

        # parse function is a generic utility function that takes a generator output and returns a dictionary of attributes 
        try: 
            parsed_output = self.config.output_class()
            attrs = self.config.output_parser(generator_output) # TODO output parser call can be made asynchronous 
            # set attributes on the parsed output object to the class 
            for attr, value in attrs.items():
                setattr(parsed_output, attr, value)

            # TODO optionally execute tools based on the parsed attributes 

        except Exception as e:
            log.error(f"Failed to parse and apply post process functions on generator output: {e}")
            # default to returning an object of the generator type with error e
            return GeneratorOutput(error=str(e), id=id)

        return parsed_output

    
    def stream(
        self,
        user_query: str, 
        current_objective: Optional[str] = None,
        memory: Optional[str] = None,
    ) -> Generator["StreamChunk", None, None]:
        """
        Synchronously executes the agent output and stream results.
        Optionally parse and post-process each chunk.
        """
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
    ) -> Generator[StreamChunk, None, None]:
        """
        Execute the agent asynchronously and stream results.
        Optionally parse and post-process each chunk.
        """
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
        return self.agent.generator.backward(
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
        stream_parser: Optional[StreamParser] = None,
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
            