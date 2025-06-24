"""WebThinker processor components."""

from typing import Optional, Dict, Any, List
from adalflow.core.component import Component
from adalflow.core.generator import Generator
from adalflow.optim.parameter import Parameter
from adalflow.core.model_client import ModelClient

class InformationIntegrator(Component):
    """Trainable component for integrating new search information into reasoning chain."""
    
    def __init__(
        self,
        model_client: Optional[ModelClient] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        
        # Initialize generator for information integration
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs or {
                "model": "gpt-3.5-turbo",
                "temperature": 0.3,
                "max_tokens": 1000,
            },
            **kwargs
        )
        
        # Trainable integration template
        self.integration_template = Parameter(
            data="""Integrate the new information from web search into your ongoing reasoning.

Previous reasoning:
{previous_reasoning}

New information from search:
{new_information}

Original question:
{question}

Continue your reasoning by incorporating this new information:""",
            requires_opt=True,
            role_desc="Template for information integration"
        )
    
    def forward(self, previous_reasoning: str, new_information: str, question: str) -> Parameter:
        """Integrate new information into reasoning chain."""
        prompt = self.integration_template.data.format(
            previous_reasoning=previous_reasoning,
            new_information=new_information,
            question=question
        )
        
        return self.generator(prompt=prompt)
    
    def call(self, previous_reasoning: str, new_information: str, question: str) -> Parameter:
        """Call the information integrator."""
        return self.forward(previous_reasoning, new_information, question)


class SearchResultProcessor(Component):
    """Trainable component for processing and formatting search results."""
    
    def __init__(
        self,
        model_client: Optional[ModelClient] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        
        # Initialize generator for result processing
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs or {
                "model": "gpt-3.5-turbo",
                "temperature": 0.1,
                "max_tokens": 600,
            },
            **kwargs
        )
        
        # Trainable processing template
        self.processing_template = Parameter(
            data="""Process and summarize these search results for the given query.

Search query: {query}

Search results:
{results}

Provide a concise summary of the most relevant information:""",
            requires_opt=True,
            role_desc="Template for search result processing"
        )
    
    def forward(self, query: str, results: str) -> Parameter:
        """Process and summarize search results."""
        prompt = self.processing_template.data.format(
            query=query,
            results=results[:2000]  # Truncate if too long
        )
        
        return self.generator(prompt=prompt)
    
    def call(self, query: str, results: str) -> Parameter:
        """Call the search result processor."""
        return self.forward(query, results)


class ContextManager(Component):
    """Component for managing and truncating reasoning context to fit within token limits."""
    
    def __init__(self, max_context_length: int = 3000):
        super().__init__()
        self.max_context_length = max_context_length
    
    def forward(self, full_context: str) -> str:
        """Manage context length by intelligent truncation."""
        if len(full_context) <= self.max_context_length:
            return full_context
        
        # Keep the beginning and end of context
        keep_start = self.max_context_length // 3
        keep_end = self.max_context_length // 3
        
        start_part = full_context[:keep_start]
        end_part = full_context[-keep_end:]
        
        truncated_context = f"{start_part}\n\n... [reasoning truncated] ...\n\n{end_part}"
        return truncated_context
    
    def call(self, full_context: str) -> str:
        """Call the context manager."""
        return self.forward(full_context)


class HistoryTracker(Component):
    """Component for tracking search history and preventing duplicate searches."""
    
    def __init__(self):
        super().__init__()
        self.search_history: List[str] = []
        self.search_results: Dict[str, str] = {}
    
    def add_search(self, query: str, results: str) -> None:
        """Add a search query and its results to history."""
        if query not in self.search_history:
            self.search_history.append(query)
            self.search_results[query] = results
    
    def has_searched(self, query: str) -> bool:
        """Check if a query has already been searched."""
        return query in self.search_history
    
    def get_previous_results(self, query: str) -> Optional[str]:
        """Get results from a previous search."""
        return self.search_results.get(query)
    
    def get_search_summary(self) -> str:
        """Get a summary of all searches performed."""
        if not self.search_history:
            return "No searches performed yet."
        
        summary = f"Searches performed ({len(self.search_history)}):\n"
        for i, query in enumerate(self.search_history, 1):
            summary += f"{i}. {query}\n"
        
        return summary
    
    def reset(self) -> None:
        """Reset search history."""
        self.search_history.clear()
        self.search_results.clear()
    
    def forward(self, query: str) -> Dict[str, Any]:
        """Check search status and return information."""
        return {
            "has_searched": self.has_searched(query),
            "previous_results": self.get_previous_results(query),
            "search_count": len(self.search_history),
            "search_history": self.search_history.copy()
        }
    
    def call(self, query: str) -> Dict[str, Any]:
        """Call the history tracker."""
        return self.forward(query)
