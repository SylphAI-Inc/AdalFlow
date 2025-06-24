"""Search engine implementations for web search tools."""

import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Individual search result."""
    title: str
    url: str
    snippet: str
    position: int = 0
    raw_data: Optional[Dict[str, Any]] = None

@dataclass 
class SearchEngineConfig:
    """Configuration for search engines."""
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

class SerperSearch:
    """Google Serper API search implementation."""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[SearchEngineConfig] = None):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        self.config = config or SearchEngineConfig()
        self.base_url = "https://google.serper.dev/search"
        
        if not self.api_key:
            raise ValueError("SERPER_API_KEY not found in environment variables or config")
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Perform search using Serper API."""
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            "q": query,
            "num": min(max(1, num_results), 10)
        }
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.config.max_retries):
                try:
                    async with session.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return self._parse_serper_results(data, query)
                        else:
                            error_text = await response.text()
                            raise aiohttp.ClientError(f"HTTP {response.status}: {error_text}")
                            
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt == self.config.max_retries - 1:
                        raise RuntimeError(f"Search failed after {self.config.max_retries} attempts: {str(e)}")
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        return []
    
    def _parse_serper_results(self, data: Dict[str, Any], query: str) -> List[SearchResult]:
        """Parse Serper API response into SearchResult objects."""
        results = []
        
        organic_results = data.get('organic', [])
        for i, item in enumerate(organic_results):
            result = SearchResult(
                title=item.get('title', ''),
                url=item.get('link', ''),
                snippet=item.get('snippet', ''),
                position=i + 1,
                raw_data=item
            )
            results.append(result)
        
        return results

class BingSearch:
    """Bing Web Search API implementation."""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[SearchEngineConfig] = None):
        self.api_key = api_key or os.getenv("BING_SEARCH_API_KEY")
        self.config = config or SearchEngineConfig()
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"
        
        if not self.api_key:
            raise ValueError("BING_SEARCH_API_KEY not found in environment variables or config")
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Perform search using Bing Web Search API."""
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        
        params = {
            'q': query,
            'count': min(max(1, num_results), 50),
            'responseFilter': 'Webpages'
        }
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.config.max_retries):
                try:
                    async with session.get(
                        self.base_url,
                        headers=headers,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return self._parse_bing_results(data, query)
                        else:
                            error_text = await response.text()
                            raise aiohttp.ClientError(f"HTTP {response.status}: {error_text}")
                            
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt == self.config.max_retries - 1:
                        raise RuntimeError(f"Search failed after {self.config.max_retries} attempts: {str(e)}")
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        return []
    
    def _parse_bing_results(self, data: Dict[str, Any], query: str) -> List[SearchResult]:
        """Parse Bing API response into SearchResult objects."""
        results = []
        
        web_pages = data.get('webPages', {}).get('value', [])
        for i, item in enumerate(web_pages):
            result = SearchResult(
                title=item.get('name', ''),
                url=item.get('url', ''),
                snippet=item.get('snippet', ''),
                position=i + 1,
                raw_data=item
            )
            results.append(result)
        
        return results

def create_search_engine(engine_type: str, api_key: Optional[str] = None, config: Optional[SearchEngineConfig] = None):
    """Factory function to create search engine instances."""
    if engine_type.lower() == "serper":
        return SerperSearch(api_key=api_key, config=config)
    elif engine_type.lower() == "bing":
        return BingSearch(api_key=api_key, config=config)
    else:
        raise ValueError(f"Unsupported search engine: {engine_type}")

async def search_web(
    query: str, 
    engine: str = "serper", 
    num_results: int = 5,
    api_key: Optional[str] = None,
    config: Optional[SearchEngineConfig] = None
) -> List[SearchResult]:
    """Unified web search function."""
    search_engine = create_search_engine(engine, api_key, config)
    return await search_engine.search(query, num_results)
