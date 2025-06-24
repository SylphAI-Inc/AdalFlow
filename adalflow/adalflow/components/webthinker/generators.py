"""WebThinker generator components."""

from typing import Optional, Dict, Any
from adalflow.core.component import Component
from adalflow.core.generator import Generator
from adalflow.optim.parameter import Parameter
from adalflow.core.model_client import ModelClient

class SearchQueryGenerator(Component):
    """Trainable component for generating search queries based on reasoning context."""
    
    def __init__(
        self,
        model_client: Optional[ModelClient] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        
        # Initialize generator with model configuration
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs or {
                "model": "gpt-3.5-turbo",
                "temperature": 0.3,
                "max_tokens": 100,
            },
            **kwargs
        )
        
        # Trainable prompt template
        self.query_template = Parameter(
            data="""Based on the current reasoning and question, generate a specific search query for missing information.

Reasoning context: {context}
Question: {question}

Search query:""",
            requires_opt=True,
            role_desc="Template for search query generation"
        )
    
    def forward(self, reasoning_context: str, question: str) -> Parameter:
        """Generate search query based on current reasoning context."""
        prompt = self.query_template.data.format(
            context=reasoning_context[:2000],  # Truncate context if too long
            question=question
        )
        
        return self.generator(prompt=prompt)
    
    def call(self, reasoning_context: str, question: str) -> Parameter:
        """Call the generator."""
        return self.forward(reasoning_context, question)


class AnswerSynthesizer(Component):
    """Trainable component for synthesizing final answers from complete reasoning chains."""
    
    def __init__(
        self,
        model_client: Optional[ModelClient] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        
        # Initialize generator with model configuration
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs or {
                "model": "gpt-3.5-turbo",
                "temperature": 0.1,
                "max_tokens": 500,
            },
            **kwargs
        )
        
        # Trainable synthesis template
        self.synthesis_template = Parameter(
            data="""Based on the complete reasoning chain, provide a comprehensive final answer.

Reasoning chain: {reasoning}
Original question: {question}

Final Answer:""",
            requires_opt=True,
            role_desc="Template for answer synthesis"
        )
    
    def forward(self, reasoning_chain: str, question: str) -> Parameter:
        """Synthesize final answer from complete reasoning chain."""
        prompt = self.synthesis_template.data.format(
            reasoning=reasoning_chain,
            question=question
        )
        
        return self.generator(prompt=prompt)
    
    def call(self, reasoning_chain: str, question: str) -> Parameter:
        """Call the synthesizer."""
        return self.forward(reasoning_chain, question)


class ReasoningContinuer(Component):
    """Trainable component for continuing reasoning without additional search."""
    
    def __init__(
        self,
        model_client: Optional[ModelClient] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        
        # Initialize generator with model configuration
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs or {
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1000,
            },
            **kwargs
        )
        
        # Trainable reasoning template
        self.reasoning_template = Parameter(
            data="""Continue your reasoning to answer the question. Think step by step.

Question: {question}
Current reasoning: {current_reasoning}

Continue reasoning:""",
            requires_opt=True,
            role_desc="Template for reasoning continuation"
        )
    
    def forward(self, question: str, current_reasoning: str = "") -> Parameter:
        """Continue reasoning based on current state."""
        prompt = self.reasoning_template.data.format(
            question=question,
            current_reasoning=current_reasoning
        )
        
        return self.generator(prompt=prompt)
    
    def call(self, question: str, current_reasoning: str = "") -> Parameter:
        """Call the reasoning continuer."""
        return self.forward(question, current_reasoning)
