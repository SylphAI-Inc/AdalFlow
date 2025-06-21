from typing import Dict, List, Optional, Type, Any
from adal import Component
from adalflow.core.model_client import ModelClient
from adalflow.core.generator import Generator
from adalflow.core.tool_manager import ToolManager
from adalflow.core.prompt import Prompt  # Assuming Prompt is defined in this module

__all__ = ["Agent"] 

class Agent(Component):
    """
    A custom agent class that orchestrates generation and tool usage.
    
    The Agent class holds a Generator and a ToolManager
    for tool usage, and system_prompt and other internal configurations that will be used by the Runner 

    It also stores convenient methods for storing the Agent as a compact representation using load_state_dict and creating the Agent 
    from a compact representation using from_config
    
    Attributes:
        name (str): Name of the agent
        tool_manager (ToolManager): Stores and manages tools
        generator (Generator): Handles text generation with a language model
        system_prompt (str): The system prompt template for the agent
        current_mode (Optional[str]): Current operational mode of the agent
    """
    def __init__(self, name: str, tool_manager: ToolManager, system_prompt: str,
                 # Generator parameters
                 # TODO in the future only require config for better readability
                 model_client: Optional[ModelClient] = None,
                 model_kwargs: Optional[Dict[str, Any]] = {},
                 template: Optional[str] = None,
                 prompt_kwargs: Optional[Dict] = {},
                 output_processors: Optional[Any] = None,
                 generator_name: Optional[str] = None,
                 cache_path: Optional[str] = None,
                 use_cache: Optional[bool] = False,
                 config_generator: Optional[Dict[str, Any]] = None,
                 # Agent parameters
                 current_mode: Optional[str] = None,
                 **kwargs):
        """Initialize agent with required components.

        You can pass a config_generator to recreate the generator or manuall pass in the parameters
        
        Args:
            name: Unique identifier for the agent
            tool_manager: Initialized ToolManager instance
            system_prompt: Template string for system prompt
            
            # Generator parameters
            model_client: Initialized ModelClient instance for the generator
            model_kwargs: Model configuration parameters (e.g., model name, temperature)
            template: Optional template for the generator prompt
            prompt_kwargs: Additional prompt parameters
            output_processors: Optional output processors for the generator
            generator_name: Optional name for the generator
            cache_path: Path for caching generator outputs
            use_cache: Whether to use caching for generator outputs
            
            # Agent parameters
            current_mode: Optional initial mode for the agent
            config_generator: Optional configuration dictionary that can be used to recreate the generator
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.name = name
        self.tool_manager = tool_manager
        self.system_prompt = system_prompt
        self.current_mode = current_mode
        
        if config_generator:
            self.config_generator = config_generator
        else:
            # either config_generator or model_client (other parameters to Generator is not strictly required) must be provided
            assert(model_client is not None, "model_client must be provided if config_generator is not provided")
            self.config_generator = {
                'model_client': model_client.to_dict(),
                'model_kwargs': model_kwargs,
                'template': template,
                'prompt_kwargs': prompt_kwargs or {},
                'output_processors': output_processors,
                'name': generator_name or f"{name}_generator",
                'cache_path': cache_path,
                'use_cache': use_cache
            }

        self.generator = Generator.from_config(self.config_generator)

    def _create_prompt_kwargs(self, user_query: str, 
                           current_objective: Optional[str] = None,
                           memory: Optional[str] = None, 
                           context: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create prompt kwargs from inputs.
        
        Args:
            user_query: The user's input query
            current_objective: Optional current objective/context
            memory: Optional memory/chat history
            context: Optional list of context strings
            
        Returns:
            Dictionary of prompt arguments for the generator
        """
        task_desc = Prompt(
            self.system_prompt,
            {"current_mode": self.current_mode or "default"},
        )()

        # create prompt_kwargs from the inputs 
        self.prompt_kwargs = {
            "task_desc_str": task_desc,
            "input_str": user_query,
            "chat_history_str": memory or "",
            "current_objective": current_objective or "",
            "context_str": context or [],
        }
        return self.prompt_kwargs
        
    def update_agent(self, 
        name: Optional[str] = None,
        tool_manager: Optional[ToolManager] = None,
        generator_config: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        current_mode: Optional[str] = None, 
        parser: Optional[OutputParser] = None, 
        stream_parser: Optional[StreamParser] = None,
    ) -> None:
        """Update agent configuration components.
        
        Args:
            generator_config: New generator configuration to use for generator 
            tool_manager: New tool manager to use
            parser: New output parser to use
            system_prompt: New system prompt to use
            current_mode: New current mode to use
            stream_parser: New stream parser to use
        """
        if generator_config is not None:
            self.config_generator = generator_config 
            self.generator = Generator.from_config(generator_config)
        if tool_manager is not None:
            self.tool_manager = tool_manager
        if parser is not None:
            self.parser = parser
        if system_prompt is not None:
            self.system_prompt = system_prompt
        if current_mode is not None:
            self.current_mode = current_mode
        if stream_parser is not None:
            self.stream_parser = stream_parser
        if name is not None:
            self.name = name

    def call (
        self,
        user_query: str,
        current_objective: Optional[str] = None,
        memory: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = {},
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
    ) -> GeneratorOutput:
        """Call the generator with the given arguments."""
        prompt_kwargs = self._create_prompt_kwargs(user_query, current_objective, memory, context)

        model_kwargs = model_kwargs or {}
        return self.generator(prompt_kwargs, model_kwargs=model_kwargs, use_cache=use_cache, id=id)

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
        prompt_kwargs = self._create_prompt_kwargs(user_query, current_objective, memory, context)

        model_kwargs = model_kwargs or {}
        return self.generator.acall(prompt_kwargs, model_kwargs=model_kwargs, use_cache=use_cache, id=id)
        
    def get_prompt(self, **kwargs) -> str:
        """Get formatted prompt using generator's prompt template.
        
        Args:
            **kwargs: Additional arguments to pass to the generator's get_prompt
            
        Returns:
            Formatted prompt string
        """
        return self.generator.get_prompt(**kwargs)
        
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools from tool manager.
        
        Returns:
            List of tool dictionaries with name, description, and schema
        """
        return self.tool_manager.get_all_tools()
        
    @classmethod
    def from_config(cls: Type['Agent'], config: Dict[str, Any]) -> 'Agent':
        """Create an Agent instance from configuration dictionary.
        
        Example:
            # Using direct configuration
            config = {
                "name": "example_agent",
                "model_client": {
                    "component_name": "OpenAIClient",
                    "component_config": {}
                },
                "model_kwargs": {"model": "gpt-3.5-turbo", "temperature": 0},
                "template": "Your custom template",
                "prompt_kwargs": {},
                "output_processors": None,
                "generator_name": "example_generator",
                "tool_manager": {
                    "tools": []
                },
                "system_prompt": "You are a helpful assistant.",
                "current_mode": "default"
            }
            
            # OR using config_generator
            config = {
                "name": "example_agent",
                "config_generator": {
                    "model_client": {
                        "component_name": "OpenAIClient",
                        "component_config": {}
                    },
                    "model_kwargs": {"model": "gpt-3.5-turbo"},
                    "template": "Your custom template",
                    "prompt_kwargs": {},
                    "output_processors": None,
                    "name": "example_generator",
                    "cache_path": None,
                    "use_cache": False
                },
                "tool_manager": {
                    "tools": []
                },
                "system_prompt": "You are a helpful assistant."
            }
            
            agent = Agent.from_config(config)
            
        Args:
            config: Configuration dictionary that may contain either individual parameters
                or a complete 'config_generator' dictionary.
                
        Returns:
            Initialized Agent instance
        """
        required_keys = {"name", "tool_manager", "system_prompt"}
        
        # If using config_generator, we don't need model_client and model_kwargs in the main config
        if "config_generator" not in config:
            required_keys.update({"model_client"}) # only strictly required parameter 
        
        missing_keys = required_keys - set(config.keys())
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
            
        # Import here to avoid circular imports
        from adalflow.core.model_client import ModelClient
        
        # use config_generator as default for the generator parameters otherwise 
        # use the individual parameters that should be stored in config 

        base_args = {
            "name": config["name"],
            "tool_manager": ToolManager.from_config(config["tool_manager"]),
            "system_prompt": config["system_prompt"],
            "current_mode": config.get("current_mode", None)
        }

        # by default if both config_generator and other parameters are provided, 
        # config_generator will be used

        if "config_generator" in config: 
            return cls(
                **base_args,
                config_generator=config["config_generator"],
            )

        return cls(
            **base_args,
            # Generator parameters
            model_client=ModelClient.from_config(config["model_client"]),
            model_kwargs=config.get("model_kwargs", {}),
            template=config.get("template", None),
            prompt_kwargs=config.get("prompt_kwargs", {}),
            output_processors=config.get("output_processors", None),
            generator_name=config.get("generator_name", None),
            cache_path=config.get("cache_path", None),
            use_cache=config.get("use_cache", False),
        )

    def return_state_dict(self, save: bool = False) -> Dict[str, Any]:
        """Return serializable state dictionary for saving/loading.
        
        Args:
            save: If True, includes additional information needed for saving
            
        Returns:
            Dictionary containing the agent's state
        """

        # any properly initialized Agent must internally store the self.config_generator
        assert(self.config_generator is not None, "config_generator should have been initialized by __init__")

        state = {
            "name": self.name,
            "tool_manager": self.tool_manager.to_dict(save=save),
            "system_prompt": self.system_prompt,
            "current_mode": self.current_mode,
            "class_name": self.__class__.__name__,
            "module": self.__class__.__module__,
            "config_generator": self.config_generator
        }
            
        return state