from typing import Dict, List, Optional, Type, Any
from adalflow.core.component import Component
from adalflow.core.model_client import ModelClient
from adalflow.core.generator import Generator
from adalflow.core.tool_manager import ToolManager
from adalflow.core.prompt_builder import Prompt
from adalflow.core.types import GeneratorOutput, ModelType, FunctionType 

from adalflow.components.agent.react import DEFAULT_REACT_AGENT_SYSTEM_PROMPT, react_agent_task_desc



__all__ = ["Agent"] 

class Agent(Component):
    """
    A custom agent class that orchestrates generation and tool usage.
    
    The Agent class holds a Generator and a ToolManager for tool usage, and other configurations for the agent and Generator 

    It also stores convenient methods for storing the Agent as a compact representation using load_state_dict and creating the Agent 
    from a compact representation using from_config
    
    Attributes:
        name (str): Name of the agent
        tool_manager (ToolManager): Stores and manages tools
        generator (Generator): Handles text generation with a language model
        the output_processors must return the type StepOutput 
    """
    def __init__(self, name: str, 
                 # Tool manager parameters
                 tools: Optional[List[Any]] = None,
                 additional_context: Optional[Dict[str, Any]] = None,
                 # Generator parameters
                 model_client: Optional[ModelClient] = None,
                 model_kwargs: Optional[Dict[str, Any]] = {},
                 model_type: Optional[ModelType] = ModelType.LLM,
                 template: Optional[str] = None ,
                 task_desc: Optional[str] = None,
                 prompt_kwargs: Optional[Dict] = {},
                 output_processors: Optional[Any] = None, 
                 generator_name: Optional[str] = None,
                 cache_path: Optional[str] = None,
                 use_cache: Optional[bool] = False,
                 config_generator: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """Initialize agent with required components.

        You can pass a config_generator to recreate the generator or manually pass in the parameters
        
        Args:
            name: Unique identifier for the agent
            
            # Tool manager parameters
            tools: List of tools to be managed by the ToolManager
            additional_context: Additional context to be added to the tool manager
            
            # Generator parameters
            model_client: Initialized ModelClient instance for the generator
            model_kwargs: Model configuration parameters (e.g., model name, temperature)
            template: Optional template for the generator prompt
            prompt_kwargs: Additional prompt parameters (The prompt_kwargs must incldue details on )
            output_processors: Optional output processors for the generator (should haeve Function as the output type) 
            generator_name: Optional name for the generator
            cache_path: Path for caching generator outputs
            use_cache: Whether to use caching for generator outputs
            
            # Agent parameters
            config_generator: Optional configuration dictionary that can be used to recreate the generator
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.name = name
        self.tool_manager = ToolManager(tools=tools or [], additional_context=additional_context or {})

        template = template or DEFAULT_REACT_AGENT_SYSTEM_PROMPT
        task_desc = task_desc or react_agent_task_desc


        prompt_kwargs = prompt_kwargs or {}
        prompt_kwargs.update({"task_desc": task_desc})

        if config_generator:
            self.config_generator = config_generator
        else:
            # either config_generator or model_client (other parameters to Generator is not strictly required) must be provided
            assert(model_client is not None, "model_client must be provided if config_generator is not provided")
            self.config_generator = {
                'model_client': model_client,
                'model_kwargs': model_kwargs,
                'template': template,
                'prompt_kwargs': prompt_kwargs,
                'output_processors': output_processors,
                'name': generator_name,
                'cache_path': cache_path,
                'use_cache': use_cache
            }

        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=template,
            prompt_kwargs=prompt_kwargs,
            output_processors=output_processors,
            name=generator_name,
            cache_path=cache_path,
            use_cache=use_cache
        )

    @property
    def template(self):
        return self._template
    
        
    # TODO need to fix from_config method 
    def update_agent(
        self,
        *,
        # ---- generic agent updates ----
        name: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        # ---- generator updates (either pass a full config OR the individual fields below) ----
        generator_config: Optional[Dict[str, Any]] = None,
        model_client: Optional[Any] = None,
        model_type: Optional[ModelType] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        template: Optional[str] = None,
        prompt_kwargs: Optional[Dict[str, Any]] = None,
        output_processors: Optional[Any] = None,
        generator_name: Optional[str] = None,
        cache_path: Optional[str] = None,
        use_cache: Optional[bool] = None,
    ) -> None:
        """Update agent configuration components.
        
        Args:
            name: New name for the agent
            tools: New list of tools to be managed by the ToolManager
            additional_context: New additional context for the ToolManager
            generator_config: New generator configuration to use for generator 
        """
        # ------------------------------------------------------------------
        # Handle generator updates
        # ------------------------------------------------------------------
        if generator_config is not None:
            # Highest-precedence: explicit full config
            self.config_generator = generator_config
            self.generator = Generator.from_config(generator_config)
        else:
            # If any individual generator field is supplied, patch the existing
            # config and rebuild the generator.
            individual_fields_provided = any(
                x is not None
                for x in (
                    model_client,
                    model_kwargs,
                    template,
                    prompt_kwargs,
                    output_processors,
                    generator_name,
                    cache_path,
                    use_cache,
                )
            )
            if individual_fields_provided:
                # Start with current config as base so unspecified fields persist
                new_cfg = dict(self.config_generator)

                if model_client is not None:
                    # ModelClient instance -> dict if possible
                    new_cfg["model_client"] = model_client
                if model_kwargs is not None:
                    new_cfg["model_kwargs"] = model_kwargs
                if template is not None:
                    new_cfg["template"] = template
                if prompt_kwargs is not None:
                    new_cfg["prompt_kwargs"] = prompt_kwargs
                if output_processors is not None:
                    new_cfg["output_processors"] = output_processors
                if generator_name is not None:
                    new_cfg["name"] = generator_name
                if cache_path is not None:
                    new_cfg["cache_path"] = cache_path
                if use_cache is not None:
                    new_cfg["use_cache"] = use_cache
                if model_type is not None:
                    new_cfg["model_type"] = model_type

                # Re-instantiate generator
                self.config_generator = new_cfg
                self.generator = Generator.from_config(new_cfg)
                # self.generator = Generator( 
                #     model_client=new_cfg["model_client"],
                #     model_kwargs=new_cfg.get("model_kwargs", {}),
                #     template=new_cfg.get("template"),
                #     prompt_kwargs=new_cfg.get("prompt_kwargs", {}),
                #     output_processors=new_cfg.get("output_processors"),
                #     generator_name=new_cfg.get("generator_name"),
                #     cache_path=new_cfg.get("cache_path"),
                #     use_cache=new_cfg.get("use_cache", False),
                # )
        if tools is not None or additional_context is not None:
            # Update tools and/or additional_context if either is provided
            current_tools = tools if tools is not None else self.tool_manager.tools
            current_context = additional_context if additional_context is not None else self.tool_manager._additional_context
            self.tool_manager = ToolManager(tools=current_tools, additional_context=current_context)
        if name is not None:
            self.name = name

    def call(
        self,
        prompt_kwargs: Dict[str, Any],
        model_kwargs: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
    ) -> GeneratorOutput:
        """Call the generator with the given arguments.
        
        Args:
            prompt_kwargs: Dictionary of prompt arguments for the generator
            model_kwargs: Optional model parameters to override defaults
            use_cache: Whether to use cached results if available
            id: Optional unique identifier for the request
            
        Returns:
            The generator output
        """
        return self.generator(prompt_kwargs, model_kwargs=model_kwargs, use_cache=use_cache, id=id)

    async def acall(
        self,
        prompt_kwargs: Dict[str, Any],
        model_kwargs: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
        id: Optional[str] = None,
    ) -> GeneratorOutput:
        """Call the generator with the given arguments asynchronously.
        
        Args:
            prompt_kwargs: Dictionary of prompt arguments for the generator
            model_kwargs: Optional model parameters to override defaults
            use_cache: Whether to use cached results if available
            id: Optional unique identifier for the request
            
        Returns:
            The generator output
        """

        return await self.generator.acall(prompt_kwargs, model_kwargs=model_kwargs, use_cache=use_cache, id=id)
    
    def is_training(self) -> bool:
        return self.generator.training
        
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
        

    # TODO fix errors with from_config 
    # @classmethod
    # def from_config(cls: Type['Agent'], config: Dict[str, Any]) -> 'Agent':
    #     """Create an Agent instance from configuration dictionary.
        
    #     Example:
    #         # Using direct configuration
    #         config = {
    #             "name": "example_agent",
    #             "model_client": {
    #                 "component_name": "OpenAIClient",
    #                 "component_config": {}
    #             },
    #             "model_kwargs": {"model": "gpt-3.5-turbo", "temperature": 0},
    #             "template": "Your custom template",
    #             "prompt_kwargs": {},
    #             "output_processors": None,
    #             "generator_name": "example_generator",
    #             "tools": [
    #                 # List of tool configurations or tool instances
    #             ],
    #             "additional_context": {
    #                 # Additional context for the tool manager
    #             },
    #             "current_mode": "default"
    #         }
            
    #         # OR using config_generator
    #         config = {
    #             "name": "example_agent",
    #             "config_generator": {
    #                 "model_client": {
    #                     "component_name": "OpenAIClient",
    #                     "component_config": {}
    #                 },
    #                 "model_kwargs": {"model": "gpt-3.5-turbo"},
    #                 "template": "Your custom template",
    #                 "prompt_kwargs": {},
    #                 "output_processors": None,
    #                 "name": "example_generator",
    #                 "cache_path": None,
    #                 "use_cache": False
    #             },
    #             "tools": [
    #                 # List of tool configurations or tool instances
    #             ],
    #             "additional_context": {
    #                 # Additional context for the tool manager
    #             },
    #             "current_mode": "default"
    #         }
            
    #         agent = Agent.from_config(config)
            
    #     Args:
    #         config: Configuration dictionary that may contain either individual parameters
    #             or a complete 'config_generator' dictionary.
                
    #     Returns:
    #         Initialized Agent instance
    #     """
    #     required_keys = {"name", "system_prompt"}
        
    #     # If using config_generator, we don't need model_client and model_kwargs in the main config
    #     if "config_generator" not in config:
    #         required_keys.update({"model_client"}) # only strictly required parameter 
        
    #     missing_keys = required_keys - set(config.keys())
    #     if missing_keys:
    #         raise ValueError(f"Missing required config keys: {missing_keys}")
            
    #     # Import here to avoid circular imports
    #     from adalflow.core.model_client import ModelClient
    #     from adalflow.core.func_tool import FunctionTool
        
    #     # Process tools if provided
    #     tools = []
    #     if "tools" in config:
    #         tools = [
    #             FunctionTool.from_config(tool) if isinstance(tool, dict) else tool 
    #             for tool in config["tools"]
    #         ]
            
    #     # Get additional context
    #     additional_context = config.get("additional_context", {})
        
    #     base_args = {
    #         "name": config["name"],
    #         "tools": tools,
    #         "additional_context": additional_context,
    #         "current_mode": config.get("current_mode")
    #     }

    #     # If config_generator is provided, use it
    #     if "config_generator" in config: 
    #         return cls(
    #             **base_args,
    #             config_generator=config["config_generator"],
    #         )

    #     # Otherwise use individual parameters
    #     return cls(
    #         **base_args,
    #         # Generator parameters
    #         model_client=ModelClient.from_config(config["model_client"]),
    #         model_kwargs=config.get("model_kwargs", {}),
    #         template=config.get("template"),
    #         prompt_kwargs=config.get("prompt_kwargs", {}),
    #         output_processors=config.get("output_processors"),
    #         generator_name=config.get("generator_name"),
    #         cache_path=config.get("cache_path"),
    #         use_cache=config.get("use_cache", False),
    #     )

    # def return_state_dict(self, save: bool = False) -> Dict[str, Any]:
    #     """Return serializable state dictionary for saving/loading.
        
    #     Args:
    #         save: If True, includes additional information needed for saving
            
    #     Returns:
    #         Dictionary containing the agent's state with the following structure:
    #         {
    #             "name": str,  # Name of the agent
    #             "tools": List[Dict],  # List of tool configurations
    #             "additional_context": Dict,  # Additional context for the tool manager
    #             "system_prompt": str,  # System prompt template
    #             "current_mode": Optional[str],  # Current operational mode
    #             "class_name": str,  # Name of the class
    #             "module": str,  # Module where the class is defined
    #             "config_generator": Dict  # Generator configuration
    #         }
    #     """
    #     # Any properly initialized Agent must internally store the self.config_generator
    #     if not hasattr(self, 'config_generator') or self.config_generator is None:
    #         raise ValueError("config_generator should have been initialized by __init__")

    #     state = {
    #         "name": self.name,
    #         "tools": [tool.to_dict() if hasattr(tool, 'to_dict') else str(tool) 
    #                  for tool in self.tool_manager.tools],
    #         "additional_context": self.tool_manager._additional_context,
    #         "current_mode": self.current_mode,
    #         "class_name": self.__class__.__name__,
    #         "module": self.__class__.__module__,
    #         "config_generator": self.config_generator
    #     }
            
    #     return state