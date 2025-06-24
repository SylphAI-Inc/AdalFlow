"""Web search tools for WebThinker."""

import asyncio
from typing import Dict, List, Optional, Any, Union
from adalflow.core.func_tool import FunctionTool
from adalflow.core.component import Component

from .search_engines import search_web, SearchResult, SearchEngineConfig
from .content_processor import (
    fetch_multiple_urls_async, 
    format_search_results,
    extract_relevant_snippet
)

class WebSearchProcessor(Component):
    """Trainable web search processor component."""
    
    def __init__(
        self, 
        engine: str = "serper",
        api_key: Optional[str] = None,
        config: Optional[SearchEngineConfig] = None,
        use_jina: bool = False,
        jina_api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self.engine = engine
        self.api_key = api_key
        self.config = config or SearchEngineConfig()
        self.use_jina = use_jina
        self.jina_api_key = jina_api_key
    
    async def search_and_process(
        self, 
        query: str, 
        max_results: int = 5,
        fetch_content: bool = False,
        max_snippet_length: int = 500
    ) -> str:
        """Search web and process results."""
        try:
            # Perform web search
            search_results = await search_web(
                query=query,
                engine=self.engine,
                num_results=max_results,
                api_key=self.api_key,
                config=self.config
            )
            
            if not search_results:
                return f"No search results found for query: {query}"
            
            # Convert to dict format for processing
            results_dict = []
            for result in search_results:
                result_dict = {
                    'title': result.title,
                    'url': result.url,
                    'snippet': result.snippet,
                    'position': result.position
                }
                results_dict.append(result_dict)
            
            # Fetch content if requested
            if fetch_content:
                urls = [result.url for result in search_results]
                url_contents = await fetch_multiple_urls_async(
                    urls=urls,
                    use_jina=self.use_jina,
                    jina_api_key=self.jina_api_key,
                    max_length=max_snippet_length
                )
                
                # Update snippets with fetched content
                for i, result_dict in enumerate(results_dict):
                    url = result_dict['url']
                    if url in url_contents:
                        content = url_contents[url]
                        if not content.startswith('[Error'):
                            # Extract relevant snippet from full content
                            relevant_snippet = extract_relevant_snippet(
                                content, query, max_snippet_length
                            )
                            result_dict['snippet'] = relevant_snippet
            
            # Format results
            formatted_results = format_search_results(results_dict)
            return formatted_results
            
        except Exception as e:
            return f"Error performing web search: {str(e)}"
    
    def forward(self, query: str, **kwargs) -> str:
        """Forward method for component interface."""
        return asyncio.run(self.search_and_process(query, **kwargs))
    
    def call(self, query: str, **kwargs) -> str:
        """Call method for component interface."""
        return self.forward(query, **kwargs)

class ContentAnalysisProcessor(Component):
    """Trainable content analysis processor."""
    
    def __init__(self, content_analyzer=None, **kwargs):
        super().__init__()
        self.content_analyzer = content_analyzer
    
    async def analyze_content(
        self, 
        content: str, 
        query: str, 
        intent: str = ""
    ) -> str:
        """Analyze web content using the content analyzer."""
        if self.content_analyzer:
            try:
                # Use trainable analyzer if available
                result = self.content_analyzer(
                    content=content,
                    search_query=query,
                    search_intent=intent
                )
                return result.data if hasattr(result, 'data') else str(result)
            except Exception as e:
                return f"Content analysis error: {str(e)}"
        else:
            # Simple fallback analysis
            return self._simple_content_analysis(content, query)
    
    def _simple_content_analysis(self, content: str, query: str) -> str:
        """Simple content analysis fallback."""
        if not content:
            return "No content to analyze."
        
        # Extract relevant snippet based on query
        relevant_snippet = extract_relevant_snippet(content, query, 800)
        
        return f"Relevant information:\n{relevant_snippet}"
    
    def forward(self, content: str, query: str, intent: str = "") -> str:
        """Forward method for component interface."""
        return asyncio.run(self.analyze_content(content, query, intent))
    
    def call(self, content: str, query: str, intent: str = "") -> str:
        """Call method for component interface."""
        return self.forward(content, query, intent)

# Standalone function tools

async def web_search_tool(
    query: str,
    max_results: int = 5,
    engine: str = "serper",
    fetch_content: bool = False,
    use_jina: bool = False,
    api_key: Optional[str] = None,
    jina_api_key: Optional[str] = None
) -> str:
    """
    Search the web and return formatted results.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        engine: Search engine to use ("serper" or "bing")  
        fetch_content: Whether to fetch full page content
        use_jina: Whether to use Jina API for content extraction
        api_key: API key for search engine
        jina_api_key: API key for Jina service
        
    Returns:
        Formatted search results as a string
    """
    processor = WebSearchProcessor(
        engine=engine,
        api_key=api_key,
        use_jina=use_jina,
        jina_api_key=jina_api_key
    )
    
    return await processor.search_and_process(
        query=query,
        max_results=max_results,
        fetch_content=fetch_content
    )

async def content_analysis_tool(
    content: str,
    query: str,
    intent: str = "",
    analyzer=None
) -> str:
    """
    Analyze web content for relevant information.
    
    Args:
        content: Web content to analyze
        query: Original search query
        intent: Search intent description
        analyzer: Optional content analyzer component
        
    Returns:
        Analyzed content with relevant information extracted
    """
    processor = ContentAnalysisProcessor(content_analyzer=analyzer)
    return await processor.analyze_content(content, query, intent)

# Factory functions for creating FunctionTools

def create_web_search_tool(
    engine: str = "serper",
    max_results: int = 5,
    api_key: Optional[str] = None,
    use_jina: bool = False,
    jina_api_key: Optional[str] = None
) -> FunctionTool:
    """Create a web search tool with specific configuration."""
    processor = WebSearchProcessor(
        engine=engine,
        api_key=api_key,
        use_jina=use_jina,
        jina_api_key=jina_api_key
    )
    
    async def search_func(
        query: str, 
        fetch_content: bool = False,
        max_snippet_length: int = 500
    ) -> str:
        return await processor.search_and_process(
            query=query,
            max_results=max_results,
            fetch_content=fetch_content,
            max_snippet_length=max_snippet_length
        )
    
    return FunctionTool(
        fn=search_func,
        component=processor  # Makes it trainable!
    )

def create_content_analysis_tool(component=None) -> FunctionTool:
    """Create a content analysis tool with optional trainable component."""
    processor = ContentAnalysisProcessor(content_analyzer=component)
    
    async def analysis_func(content: str, query: str, intent: str = "") -> str:
        return await processor.analyze_content(content, query, intent)
    
    return FunctionTool(
        fn=analysis_func,
        component=processor
    )
