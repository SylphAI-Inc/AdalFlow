"""WebThinker analyzer components."""

from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from adalflow.core.component import Component
from adalflow.core.types import ModelType
from adalflow.optim.parameter import Parameter
from adalflow.core.model_client import ModelClient


class SearchIntentAnalyzer(Component):
    """Trainable component for analyzing search intent from reasoning context."""
    
    def __init__(
        self,
        model_client: Optional[ModelClient] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        
        # Initialize generator for intent analysis
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs or {
                "model": "gpt-3.5-turbo",
                "temperature": 0.2,
                "max_tokens": 200,
            },
            **kwargs
        )
        
        # Trainable intent analysis template
        self.intent_template = Parameter(
            data="""Analyze the detailed intent behind this search query based on the reasoning context.

Reasoning context: {context}
Search query: {query}

What specific information is being sought? What is the detailed search intent?""",
            requires_opt=True,
            role_desc="Template for search intent analysis"
        )
    
    def forward(self, search_query: str, reasoning_context: str) -> Parameter:
        """Analyze the intent behind a search query."""
        prompt = self.intent_template.data.format(
            query=search_query,
            context=reasoning_context  # Truncate if too long
        )
        
        return self.generator(prompt=prompt)
    
    def call(self, search_query: str, reasoning_context: str) -> Parameter:
        """Call the intent analyzer."""
        return self.forward(search_query, reasoning_context)


class ContentAnalyzer(Component):
    """Trainable component for analyzing web content and extracting relevant information."""
    
    def __init__(
        self,
        model_client: Optional[ModelClient] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        
        # Initialize generator for content analysis
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs or {
                "model": "gpt-3.5-turbo",
                "temperature": 0.1,
                "max_tokens": 800,
            },
            **kwargs
        )
        
        # Trainable content analysis template
        self.analysis_template = Parameter(
            data="""Extract relevant information from the web content based on the search query and intent.

Search query: {query}
Search intent: {intent}

Web content:
{content}

Extract all relevant information that addresses the search query and intent. Focus on factual, accurate information:""",
            requires_opt=True,
            role_desc="Template for web content analysis"
        )
    
    def forward(self, content: str, search_query: str, search_intent: str) -> Parameter:
        """Analyze web content and extract relevant information."""
        # Truncate content if too long to fit in context
        # max_content_length = 3000
        # if len(content) > max_content_length:
        #     content = content[:max_content_length] + "..."
        
        prompt = self.analysis_template.data.format(
            query=search_query,
            intent=search_intent,
            content=content
        )
        
        return self.generator(prompt=prompt)
    
    def call(self, content: str, search_query: str, search_intent: str) -> Parameter:
        """Call the content analyzer."""
        return self.forward(content, search_query, search_intent)


class RelevanceFilter:
    """Non-LLM based relevance filter that follows WebThinker's approach."""
    
    def __init__(self):
        """Initialize the relevance filter."""
        pass
    
    def extract_relevant_info(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract relevant information from Google Serper search results.
        This follows the exact same approach as WebThinker's extract_relevant_info_serper function.

        Args:
            search_results (dict): JSON response from the Google Serper API.

        Returns:
            list: A list of dictionaries containing the extracted information.
        """
        useful_info = []
        if 'organic' in search_results:
            for i, result in enumerate(search_results['organic']):
                # Try to extract domain for site_name, or leave empty
                site_name = ''
                try:
                    site_name = urlparse(result.get('link', '')).netloc
                except Exception:
                    pass

                info = {
                    'id': i + 1,
                    'title': result.get('title', ''),
                    'url': result.get('link', ''),
                    'site_name': site_name,  # Serper doesn't directly provide siteName, try to parse from URL
                    'date': result.get('date', ''),  # Serper might not always provide date
                    'snippet': result.get('snippet', ''),
                    'context': ''  # Reserved field
                }
                useful_info.append(info)
        return useful_info
    
    def filter_results(self, search_results: Dict[str, Any], max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Filter search results from Serper API.
        
        Args:
            search_results: Raw search results from the Serper API
            max_results: Maximum number of results to return (None for all)
            
        Returns:
            List of filtered and structured results
        """
        results = self.extract_relevant_info(search_results)
        
        # Apply max_results limit if specified
        if max_results is not None:
            results = results[:max_results]
            
        return results


class SearchDecisionMaker(Component):
    """Trainable component for deciding whether more search is needed."""
    
    def __init__(
        self,
        model_client: Optional[ModelClient] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        
        # Initialize generator for search decisions
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs or {
                "model": "gpt-3.5-turbo",
                "temperature": 0.1,
                "max_tokens": 50,
            },
            **kwargs
        )
        
        # Trainable decision template
        self.decision_template = Parameter(
            data="""Based on the current reasoning and the question, do you need more information from web search?

Question: {question}
Current reasoning: {reasoning}

Answer with "YES" if more search is needed, "NO" if you have enough information to answer:""",
            requires_opt=True,
            role_desc="Template for search decision making"
        )
    
    def forward(self, question: str, current_reasoning: str) -> Parameter:
        """Decide if more search is needed."""
        prompt = self.decision_template.data.format(
            question=question,
            reasoning=current_reasoning[-1000:]  # Use last part of reasoning
        )
        
        return self.generator(prompt=prompt)
    
    def call(self, question: str, current_reasoning: str) -> Parameter:
        """Call the search decision maker."""
        return self.forward(question, current_reasoning)
