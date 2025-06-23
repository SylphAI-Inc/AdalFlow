import logging
from typing import Generator as GeneratorType, Dict, Optional, List, Any, Callable, Type, TypeVar, Generic, Union, Generator
from dataclasses import dataclass, field

from adalflow.core.component import Component
from adalflow.core.generator import Generator
from adalflow.core.tool_manager import ToolManager
from adalflow.components.output_parsers.outputs import OutputParser
from adalflow.core.agent import Agent
from adalflow.core.runner import Runner
from adalflow.core.types import GeneratorOutput, FunctionOutput 
from adalflow.optim.parameter import Parameter
from adalflow.core.types import Function

log = logging.getLogger(__name__)

__all__ = ["MultiRunner"]

class MultiRunner(Component):
    """
    A multirunner class that manages multiple adal.Agents through their Runners.
    
    Attributes:
        runners: Dictionary mapping unique names to adal.Runner instances
    """

    def __init__(self, runners: Dict[str, Runner], **kwargs):
        """Initialize adal.MultiRunner with a dictionary of named Runners.
        
        Args:
            runners: Dictionary mapping unique names to adal.Runner instances
        """
        super().__init__(**kwargs)
        self.runners = runners

    def add_runner(self, runner_name: str, runner: Runner) -> None:
        """Add a runner to the current dictionary of runners.
        
        Args:
            runner_name: Name for the runner
            runner: Runner instance to add
            
        Raises:
            ValueError: If a runner with the same name already exists
        """
        if runner_name in self.runners:
            raise ValueError(f"Runner with name {runner_name} already exists")
        self.runners[runner_name] = runner

    def get_runner(self, runner_name: str) -> Runner:
        """Get a runner by name.
        
        Args:
            runner_name: Name of the runner to retrieve
            
        Returns:
            The requested Runner instance
            
        Raises:
            KeyError: If no runner exists with the given name
        """
        if runner_name not in self.runners:
            raise KeyError(f"No runner found with name: {runner_name}")
        return self.runners[runner_name]

    def call(
        self,
        runner_name: str, 
        user_query: str,
        current_objective: Optional[str] = None,
        memory: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
		context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute the specified runner synchronously.
        
        Args:
            runner_name: Name of the runner to execute
            user_query: The user's input query
            current_objective: Optional current objective/context
            memory: Optional memory/chat history
            model_kwargs: Optional model-specific parameters
            use_cache: Whether to use cached responses if available
            id: Optional identifier for the request
            
        Returns:
            The output from the runner's call method
        """
        runner = self.get_runner(runner_name)
        return runner.call(
            user_query=user_query,
            current_objective=current_objective,
            memory=memory,
            model_kwargs=model_kwargs or {},
            use_cache=use_cache,
            id=id,
			context=context
        )

    async def acall(
        self,
        runner_name: str,
        user_query: str,
        current_objective: Optional[str] = None,
        memory: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
		context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute the specified runner asynchronously.
        
        Args:
            runner_name: Name of the runner to execute
            user_query: The user's input query
            current_objective: Optional current objective/context
            memory: Optional memory/chat history
            model_kwargs: Optional model-specific parameters
            use_cache: Whether to use cached responses if available
            id: Optional identifier for the request
            
        Returns:
            The output from the runner's acall method
        """
        runner = self.get_runner(runner_name)
        return await runner.acall(
            user_query=user_query,
            current_objective=current_objective,
            memory=memory,
            model_kwargs=model_kwargs or {},
            use_cache=use_cache,
            id=id,
			context=context
        )

    def stream(
        self,
        runner_name: str, 
        user_query: str, 
        current_objective: Optional[str] = None,
        memory: Optional[str] = None,
		context: Optional[Dict[str, Any]] = None
    ) -> GeneratorType[Any, None, None]:
        """
        Stream results from the specified runner synchronously.
        
        Args:
            runner_name: Name of the runner to execute
            user_query: The user's input query
            current_objective: Optional current objective/context
            memory: Optional memory/chat history
            
        Yields:
            StreamChunk objects containing the streamed output
        """
		# TODO modify Any to StreamChunk
        runner = self.get_runner(runner_name)
        yield from runner.stream(
            user_query=user_query,
            current_objective=current_objective,
            memory=memory,
			context=context
        )

    async def astream(
        self,
        runner_name: str, 
        user_query: str, 
        current_objective: Optional[str] = None,
        memory: Optional[str] = None,
		context: Optional[Dict[str, Any]] = None
    ) -> GeneratorType[Any, None, None]:
        """
        Stream results from the specified runner asynchronously.
        
        Args:
            runner_name: Name of the runner to execute
            user_query: The user's input query
            current_objective: Optional current objective/context
            memory: Optional memory/chat history
            
        Yields:
            StreamChunk objects containing the streamed output
        """

		# TODO modify Any to StreamChunk

        runner = self.get_runner(runner_name)
        async for chunk in runner.astream(
            user_query=user_query,
            current_objective=current_objective,
            memory=memory,
			context=context
        ):
            yield chunk

    def backward(
        self,
        runner_name: str,
        response: Parameter,
        prompt_kwargs: Optional[Dict] = None,
        template: Optional[str] = None,
        backward_engine: Optional[Generator] = None,
        id: Optional[str] = None,
        disable_backward_engine: bool = False
    ) -> Any:
        """
        Run backward pass on a response using the specified runner.
        
        Args:
            runner_name: Name of the runner to use
            response: The response parameter to run backward on
            prompt_kwargs: Optional prompt keyword arguments
            template: Optional template to use
            backward_engine: Optional backward engine to use
            id: Optional identifier for the request
            disable_backward_engine: Whether to disable the backward engine
            
        Returns:
            The result of the backward pass
        """
        runner = self.get_runner(runner_name)
        return runner.backward(
            response=response,
            prompt_kwargs=prompt_kwargs,
            template=template,
            backward_engine=backward_engine,
            id=id,
            disable_backward_engine=disable_backward_engine
        )

    def _tool_execute(
        self,
        runner_name: str,
        func: Union[Callable, Parameter],
        map_fn: Optional[Callable] = None,
    ) -> Any:
        """Execute a tool function through the specified runner's tool manager.

        This wrapper aligns with the current `Runner._tool_execute` signature, which
        only accepts the function to execute.  If a ``map_fn`` is provided we apply
        it to the result **after** the underlying runner call so that existing user
        code which relies on the mapping behaviour still works.

        Args:
            runner_name: Name of the runner whose tool manager to use.
            func: The function (or ``Function`` expression/``Parameter``) to execute.
            map_fn: Optional callable applied to the runner result.

        Returns:
            The raw or post-processed result of the tool execution.
        """
        runner = self.get_runner(runner_name)

        # Execute using the runner – signature now only expects ``func``
        result = runner._tool_execute(func)

        # Apply optional mapping for backwards compatibility
        return map_fn(result) if map_fn is not None else result

    def update_runner(
        self,
        runner_name: str,
        **kwargs
    ) -> None:
        """
        Update a runner's configuration.

		the *kwargs should include exactly the parameters needed by the update_runner class method of the runner.
        
        Args:
            runner_name: Name of the runner to update
            **kwargs: Configuration parameters to update
        """
        if runner_name not in self.runners:
            raise KeyError(f"No runner found with name: {runner_name}")
        
        # Update the runner's configuration
        runner = self.get_runner(runner_name)
        try: 
            runner.update_runner(**kwargs)
        except Exception as e: 
            raise ValueError(f"Failed to update runner {runner_name}: {e}")