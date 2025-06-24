"""WebThinker configuration."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class WebThinkerConfig:
    """Configuration for WebThinker components."""
    
    # Search behavior
    max_searches: int = 10
    max_tokens: int = 40000
    search_engine: str = "serper"  # "serper" or "bing"
    top_k_results: int = 5
    
    # Content processing
    use_jina: bool = True
    fetch_content: bool = False
    max_snippet_length: int = 500
    
    # Model parameters
    temperature: float = 0.7
    top_p: float = 0.8
    min_p: float = 0.05
    top_k_sampling: int = 20
    repetition_penalty: float = 1.05
    
    # API configurations
    serper_api_key: Optional[str] = None
    bing_api_key: Optional[str] = None
    jina_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # Component configurations
    search_generator_config: Dict[str, Any] = field(default_factory=dict)
    intent_analyzer_config: Dict[str, Any] = field(default_factory=dict)
    content_analyzer_config: Dict[str, Any] = field(default_factory=dict)
    integrator_config: Dict[str, Any] = field(default_factory=dict)
    synthesizer_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training parameters
    enable_training: bool = False
    learning_rate: float = 1e-4
    batch_size: int = 1
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.search_engine not in ["serper", "bing"]:
            raise ValueError(f"Unsupported search engine: {self.search_engine}")
        
        if self.max_searches <= 0:
            raise ValueError("max_searches must be positive")
        
        if self.top_k_results <= 0:
            raise ValueError("top_k_results must be positive")
