"""WebThinker Agent - Auto-differentiable web search reasoning agent."""

import re
import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass

from adalflow.core.component import Component
from adalflow.core.generator import Generator
from adalflow.core.model_client import ModelClient
from adalflow.optim.parameter import Parameter
from adalflow.components.model_client import AnthropicAPIClient
from adalflow.tools.web_search import ContentProcessor

from .config import WebThinkerConfig
from .prompts import WebThinkerPrompts
from .generators import SearchQueryGenerator, AnswerSynthesizer, ReasoningContinuer
from .analyzers import SearchIntentAnalyzer, ContentAnalyzer, SearchDecisionMaker
from .processors import InformationIntegrator, ContextManager, HistoryTracker

# Special tokens matching original WebThinker
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
BEGIN_CLICK_LINK = "<|begin_click_link|>"
END_CLICK_LINK = "<|end_click_link|>"
BEGIN_CLICK_RESULT = "<|begin_click_result|>"
END_CLICK_RESULT = "<|end_click_result|>"

# Error indicators for content validation
ERROR_INDICATORS = [
    'limit exceeded',
    'Error fetching',
    'Account balance not enough',
    'Invalid bearer token',
    'HTTP error occurred',
    'Error: Connection error occurred',
    'Error: Request timed out',
    'Unexpected error',
    'Please turn on Javascript',
    'Enable JavaScript',
    'port=443',
    'Please enable cookies',
]

# Invalid search queries to filter out
INVALID_SEARCH_QUERIES = [
    "and end with",
    "search query",
    "query",
    "your query here",
    "your query",
    "your search query",
]

log = logging.getLogger(__name__)

def extract_between(text: str, start_marker: str, end_marker: str) -> Optional[str]:
    """Extract text between two markers."""
    try:
        start_idx = text.rfind(start_marker)
        if start_idx == -1:
            return None
        start_idx += len(start_marker)
        
        end_idx = text.find(end_marker, start_idx)
        if end_idx == -1:
            return None
        
        return text[start_idx:end_idx].strip()
    except Exception:
        return None

def count_tokens(text: str) -> int:
    """Simple token counting using word splitting."""
    return len(text.split())

def has_error_indicators(content: str) -> bool:
    """Check if content contains error indicators."""
    content_lower = content.lower()
    return any(indicator.lower() in content_lower for indicator in ERROR_INDICATORS) and len(content.split()) < 64

@dataclass
class InteractionRecord:
    """Record of search or click interaction."""
    type: str  # 'search' or 'click'
    query_or_url: str
    results: str
    timestamp: float

class WebThinkerAgent(Component):
    """Auto-differentiable WebThinker agent following original implementation logic."""
    
    def __init__(
        self,
        config: Optional[WebThinkerConfig] = None,
        model_client: Optional[ModelClient] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        aux_model_client: Optional[ModelClient] = None,
        aux_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        
        # Configuration
        self.config = config or WebThinkerConfig()
        
        # Primary model for main reasoning (with interleaved thinking if Anthropic)
        if isinstance(model_client, AnthropicAPIClient):
            self.model_client = AnthropicAPIClient(
                api_key=model_client._api_key,
                support_interleaved_thinking=True
            )
        else:
            self.model_client = model_client
            
        self.model_kwargs = model_kwargs or {
            "model": "claude-sonnet-4-20250514" if isinstance(model_client, AnthropicAPIClient) else "gpt-4",
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        # Auxiliary model for intent analysis and content processing
        self.aux_model_client = aux_model_client or model_client
        self.aux_model_kwargs = aux_model_kwargs or {
            "model": "gpt-3.5-turbo",
            "temperature": 0.3,
            "max_tokens": 1000,
        }
        
        # Primary generator (with interleaved thinking support)
        self.generator = Generator(
            model_client=self.model_client,
            model_kwargs=self.model_kwargs
        )
        
        # Auxiliary generator
        self.aux_generator = Generator(
            model_client=self.aux_model_client,
            model_kwargs=self.aux_model_kwargs
        )
        
        # Initialize trainable components
        self.prompts = WebThinkerPrompts()
        self.search_intent_analyzer = SearchIntentAnalyzer(
            model_client=self.aux_model_client,
            model_kwargs=self.aux_model_kwargs
        )
        self.content_analyzer = ContentAnalyzer(
            model_client=self.aux_model_client,
            model_kwargs=self.aux_model_kwargs
        )
        
        # Content processor for URL fetching
        self.content_processor = ContentProcessor(
            use_jina=self.config.use_jina,
            jina_api_key=self.config.jina_api_key
        )
        
        # Tools (will be set externally)
        self.web_search_tool = None
        self.content_analysis_tool = None
        
        # State tracking
        self.search_cache = {}
        self.url_cache = {}
        self.executed_search_queries: Set[str] = set()
        self.clicked_urls: Set[str] = set()
        
    def set_tools(self, **tools):
        """Set the web search and content analysis tools."""
        if "web_search_tool" in tools:
            self.web_search_tool = tools["web_search_tool"]
        if "content_analysis_tool" in tools:
            self.content_analysis_tool = tools["content_analysis_tool"]
    
    def forward(self, question: str) -> Parameter:
        """Forward pass - synchronous wrapper."""
        if asyncio.iscoroutinefunction(self._forward_async):
            return asyncio.run(self._forward_async(question))
        return self._forward_async(question)
    
    async def _forward_async(self, question: str) -> Parameter:
        """
        Async implementation following the original generate_deep_web_explorer logic.
        Uses token-driven flow control with special tokens.
        """
        # Reset state for new question
        self.executed_search_queries.clear()
        self.clicked_urls.clear()
        
        # Initialize with deep web explorer instruction
        prompt = self._get_deep_web_explorer_instruction(
            search_query=question,
            search_intent="Find comprehensive information to answer the question",
            search_result=""
        )
        
        output = ""
        original_prompt = ""
        total_tokens = count_tokens(prompt)
        MAX_TOKENS = self.config.max_tokens or 30000
        MAX_INTERACTIONS = 10  # Maximum combined searches and clicks
        total_interactions = 0
        finished = False
        first_generation = True
        interaction_records: List[InteractionRecord] = []
        
        while True:
            # Generate next response with special token stops
            stop_tokens = [END_SEARCH_QUERY, END_CLICK_LINK]
            
            # Use interleaved thinking for Anthropic models
            if isinstance(self.model_client, AnthropicAPIClient):
                if "tools" not in self.model_kwargs:
                    self.model_kwargs["tools"] = self._get_anthropic_tools()
                
                response_data = await self._generate_response_anthropic(
                    prompt=prompt,
                    first_generation=first_generation,
                    stop_tokens=stop_tokens
                )
                response = response_data.get('text', [''])[0] if response_data.get('text') else ""
                formatted_prompt = prompt
            else:
                # Standard generation for other models
                formatted_prompt, response = await self._generate_response_standard(
                    prompt=prompt,
                    first_generation=first_generation,
                    stop_tokens=stop_tokens
                )
            
            if first_generation:
                original_prompt = formatted_prompt
                prompt = formatted_prompt
            
            # Clean response (remove thinking tokens like </think>)
            clean_response = response.replace('</think>\n', '')
            output += clean_response
            total_tokens = count_tokens(prompt) + count_tokens(response)
            first_generation = False
            
            # Check limits
            if total_tokens >= MAX_TOKENS or total_interactions >= MAX_INTERACTIONS:
                break
            
            # Parse response for special tokens and execute actions
            action_taken = False
            
            # Check for search query
            if response.rstrip().endswith(END_SEARCH_QUERY):
                search_query = extract_between(response, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
                if search_query and self._is_valid_search_query(search_query):
                    total_interactions += 1
                    
                    if search_query in self.executed_search_queries:
                        # Duplicate search
                        search_result = f"\n{BEGIN_SEARCH_RESULT}\nYou have already searched for this query. Please use the previously found information.\n{END_SEARCH_RESULT}\n\nOkay,"
                    else:
                        # Execute new search
                        self.executed_search_queries.add(search_query)
                        search_result = await self._execute_search(search_query)
                        
                        # Record interaction
                        interaction_records.append(InteractionRecord(
                            type='search',
                            query_or_url=search_query,
                            results=search_result,
                            timestamp=asyncio.get_event_loop().time()
                        ))
                    
                    output += search_result
                    prompt += output
                    total_tokens += count_tokens(search_result)
                    action_taken = True
            
            # Check for click link
            elif response.rstrip().endswith(END_CLICK_LINK):
                url = extract_between(response, BEGIN_CLICK_LINK, END_CLICK_LINK)
                if url:
                    total_interactions += 1
                    
                    if url in self.clicked_urls:
                        # Duplicate click
                        click_result = f"\n{BEGIN_CLICK_RESULT}\nYou have already clicked this URL.\n{END_CLICK_RESULT}\n\nOkay,"
                    else:
                        # Execute new click
                        self.clicked_urls.add(url)
                        click_result = await self._execute_click(url, output)
                        
                        # Record interaction
                        interaction_records.append(InteractionRecord(
                            type='click',
                            query_or_url=url,
                            results=click_result,
                            timestamp=asyncio.get_event_loop().time()
                        ))
                    
                    output += click_result
                    prompt += output
                    total_tokens += count_tokens(click_result)
                    action_taken = True
            
            # If no action taken, we're done
            if not action_taken:
                finished = True
                break
        
        # Add limit message if needed
        if not finished and (total_tokens >= MAX_TOKENS or total_interactions >= MAX_INTERACTIONS):
            limit_message = f"\n{BEGIN_CLICK_RESULT}\nYou have reached the interaction limit.\n{END_CLICK_RESULT}\n\nOK, I will now provide the final information based on my collected information.\n\n**Final Information:**"
            output += limit_message
            prompt += output
            
            # Generate final response
            final_response_data = await self._generate_response_standard(
                prompt=prompt,
                first_generation=False,
                max_tokens=512
            )
            final_response = final_response_data[1] if isinstance(final_response_data, tuple) else final_response_data
            output += final_response
        
        # Store interaction history
        self.interaction_records = interaction_records
        
        return Parameter(data=output)
    
    def _get_deep_web_explorer_instruction(self, search_query: str, search_intent: str, search_result: str) -> str:
        """Get the deep web explorer instruction prompt."""
        return f"""You are a reasoning assistant with web search and URL exploration capabilities. Your goal is to find comprehensive information to help answer questions accurately.

You have access to two special tools:

1. **Web Search**: To search for information, write {BEGIN_SEARCH_QUERY}your search query here{END_SEARCH_QUERY}
2. **Click URL**: To read content from a specific URL, write {BEGIN_CLICK_LINK}URL_here{END_CLICK_LINK}

**Instructions:**
- Use searches to find relevant information and sources
- Click on promising URLs to get detailed content
- Think step by step and build your understanding iteratively
- Avoid repeating the same searches or clicking the same URLs
- Provide comprehensive analysis based on the information you gather

**Question:** {search_query}

**Search Intent:** {search_intent}

{f"**Initial Search Results:** {search_result}" if search_result else ""}

Let me search for information to help answer this question comprehensively.

"""
    
    def _get_anthropic_tools(self) -> List[Dict]:
        """Get tool definitions for Anthropic models."""
        return [
            {
                "name": "web_search",
                "description": "Search the web for information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "click_url",
                "description": "Fetch and analyze content from a URL", 
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to fetch content from"
                        }
                    },
                    "required": ["url"]
                }
            }
        ]
    
    async def _generate_response_anthropic(
        self,
        prompt: str,
        first_generation: bool,
        stop_tokens: List[str]
    ) -> Dict:
        """Generate response using Anthropic's interleaved thinking."""
        try:
            kwargs = self.model_kwargs.copy()
            kwargs.update({
                "thinking": {
                    "type": "enabled", 
                    "budget_tokens": 10000
                },
                "extra_headers": {
                    "anthropic-beta": "interleaved-thinking-2025-05-14"
                }
            })
            
            result = self.generator(prompt=prompt, **kwargs)
            
            if hasattr(result, 'data') and isinstance(result.data, dict):
                return result.data
            else:
                # Fallback to text response
                return {"text": [str(result.data) if hasattr(result, 'data') else str(result)]}
                
        except Exception as e:
            log.error(f"Error in Anthropic generation: {e}")
            # Fallback to standard generation
            return await self._generate_response_standard(prompt, first_generation, stop_tokens)
    
    async def _generate_response_standard(
        self,
        prompt: str,
        first_generation: bool,
        stop_tokens: Optional[List[str]] = None,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, str]:
        """Generate response using standard model client."""
        try:
            kwargs = self.model_kwargs.copy()
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if stop_tokens:
                kwargs["stop"] = stop_tokens
                
            result = self.generator(prompt=prompt, **kwargs)
            response = result.data if hasattr(result, 'data') else str(result)
            
            return prompt, response
            
        except Exception as e:
            log.error(f"Error in standard generation: {e}")
            return prompt, f"Error generating response: {str(e)}"
    
    def _is_valid_search_query(self, query: str) -> bool:
        """Check if search query is valid."""
        if not query or len(query) <= 5:
            return False
        if END_SEARCH_QUERY in query:
            return False
        if query.lower() in [q.lower() for q in INVALID_SEARCH_QUERIES]:
            return False
        return True
    
    async def _execute_search(self, search_query: str) -> str:
        """Execute web search and return formatted results."""
        try:
            if not self.web_search_tool:
                return f"\n{BEGIN_SEARCH_RESULT}\nWeb search tool not available.\n{END_SEARCH_RESULT}\n\n"
            
            # Check cache first
            if search_query in self.search_cache:
                results = self.search_cache[search_query]
            else:
                # Perform search
                search_results = await self.web_search_tool(search_query)
                results = search_results if isinstance(search_results, str) else str(search_results)
                self.search_cache[search_query] = results
            
            log.info(f'Searched for: "{search_query}"')
            
            return f"\n{BEGIN_SEARCH_RESULT}\n{results}\n{END_SEARCH_RESULT}\n\n"
            
        except Exception as e:
            log.error(f"Error executing search for '{search_query}': {e}")
            return f"\n{BEGIN_SEARCH_RESULT}\nError performing search: {str(e)}\n{END_SEARCH_RESULT}\n\n"
    
    async def _execute_click(self, url: str, context: str) -> str:
        """Execute URL click and return formatted results."""
        try:
            # Generate click intent
            click_intent_prompt = f"""Based on the following conversation context, what specific information are we looking for from this URL?

Context: {context[-2000:]}  # Last 2000 chars

URL to click: {url}

Provide a brief description of what information we're seeking from this URL:"""
            
            intent_result = self.aux_generator(prompt=click_intent_prompt)
            click_intent = intent_result.data if hasattr(intent_result, 'data') else str(intent_result)
            
            log.info(f"Clicking on URL: {url} with intent: {click_intent}")
            
            # Fetch content
            if url not in self.url_cache:
                try:
                    content = await self.content_processor.fetch_content_async([url])
                    content = content.get(url, "")
                    
                    # Check for errors
                    if not has_error_indicators(content) and content:
                        self.url_cache[url] = content
                    else:
                        content = "Unable to fetch the page content. You can try other links."
                except Exception as e:
                    content = "Unable to fetch the page content. You can try other links."
            else:
                content = self.url_cache[url]
            
            # Check if content has errors
            if has_error_indicators(content) or not content:
                summary = "Unable to fetch the page content. You can try other links."
            else:
                # Use content analyzer to summarize
                try:
                    analysis_result = self.content_analyzer(
                        content=content[:5000],  # Limit content length
                        search_query=url,
                        search_intent=click_intent
                    )
                    summary = analysis_result.data if hasattr(analysis_result, 'data') else str(analysis_result)
                except Exception as e:
                    log.error(f"Error analyzing content: {e}")
                    summary = content[:1000] + "..." if len(content) > 1000 else content
            
            return f"\n{BEGIN_CLICK_RESULT}\n{summary}\n{END_CLICK_RESULT}\n"
            
        except Exception as e:
            log.error(f"Error executing click for '{url}': {e}")
            return f"\n{BEGIN_CLICK_RESULT}\nError fetching URL content: {str(e)}\n{END_CLICK_RESULT}\n"
    
    def call(self, question: str) -> Parameter:
        """Call the agent with a question."""
        return self.forward(question)
    
    def get_interaction_history(self) -> List[InteractionRecord]:
        """Get the interaction history for the current session."""
        return getattr(self, 'interaction_records', [])
    
    def get_search_history(self) -> Dict[str, Any]:
        """Get search history summary."""
        records = self.get_interaction_history()
        searches = [r for r in records if r.type == 'search']
        clicks = [r for r in records if r.type == 'click']
        
        return {
            "total_interactions": len(records),
            "search_count": len(searches),
            "click_count": len(clicks),
            "searches": [r.query_or_url for r in searches],
            "clicks": [r.query_or_url for r in clicks],
            "interaction_records": records
        }
