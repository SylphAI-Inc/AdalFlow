"""Content processing utilities for web search."""

import re
import asyncio
import aiohttp
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, NavigableString
import logging

log = logging.getLogger(__name__)

# Common error indicators in web content
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
    'Access denied',
    'Forbidden',
    'Not found',
    '404',
    '403',
    '500',
]

class ContentProcessor:
    """Processor for web content extraction and cleaning."""
    
    def __init__(self, use_jina: bool = False, jina_api_key: Optional[str] = None):
        self.use_jina = use_jina
        self.jina_api_key = jina_api_key
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def process_url(self, url: str, max_length: int = 3000) -> str:
        """Process a single URL and extract content."""
        try:
            if self.use_jina and self.jina_api_key:
                content = await self._fetch_with_jina(url)
            else:
                content = await self._fetch_with_requests(url)
            
            # Clean and truncate content
            cleaned_content = self._clean_content(content)
            
            # Check for error indicators
            if self._has_error_indicators(cleaned_content):
                return f"[Error: Unable to access content from {url}]"
            
            # Truncate if too long
            if len(cleaned_content) > max_length:
                cleaned_content = cleaned_content[:max_length] + "..."
            
            return cleaned_content
            
        except Exception as e:
            log.error(f"Error processing URL {url}: {str(e)}")
            return f"[Error processing {url}: {str(e)}]"
    
    async def _fetch_with_jina(self, url: str) -> str:
        """Fetch content using Jina API."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            'Authorization': f'Bearer {self.jina_api_key}',
            'Accept': 'application/json'
        }
        
        try:
            async with self.session.get(jina_url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('content', '')
                else:
                    error_text = await response.text()
                    raise aiohttp.ClientError(f"Jina API error: {response.status} - {error_text}")
        except Exception as e:
            log.warning(f"Jina API failed for {url}, falling back to direct fetch: {str(e)}")
            return await self._fetch_with_requests(url)
    
    async def _fetch_with_requests(self, url: str) -> str:
        """Fetch content using direct HTTP requests."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            async with self.session.get(url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' in content_type:
                        html_content = await response.text()
                        return extract_text_from_html(html_content)
                    else:
                        return f"[Non-HTML content: {content_type}]"
                else:
                    return f"[HTTP Error {response.status}: {response.reason}]"
        except Exception as e:
            return f"[Error fetching {url}: {str(e)}]"
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        if not content:
            return ""
        
        # Remove extra whitespace and newlines
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Remove non-printable characters
        content = re.sub(r'[^\x20-\x7E\n]', ' ', content)
        
        # Remove common navigation elements
        content = re.sub(r'(Home|About|Contact|Privacy|Terms|Menu|Navigation|Footer|Header)', '', content, flags=re.IGNORECASE)
        
        return content
    
    def _has_error_indicators(self, content: str) -> bool:
        """Check if content contains error indicators."""
        content_lower = content.lower()
        return any(indicator.lower() in content_lower for indicator in ERROR_INDICATORS)

def extract_text_from_html(html_content: str) -> str:
    """Extract text from HTML content using BeautifulSoup."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    except Exception as e:
        log.error(f"Error extracting text from HTML: {str(e)}")
        return f"[Error processing HTML content: {str(e)}]"

async def fetch_url_content_async(
    url: str, 
    use_jina: bool = False, 
    jina_api_key: Optional[str] = None,
    max_length: int = 3000
) -> str:
    """Fetch content from a single URL asynchronously."""
    async with ContentProcessor(use_jina=use_jina, jina_api_key=jina_api_key) as processor:
        return await processor.process_url(url, max_length)

async def fetch_multiple_urls_async(
    urls: List[str], 
    use_jina: bool = False, 
    jina_api_key: Optional[str] = None,
    max_length: int = 3000
) -> Dict[str, str]:
    """Fetch content from multiple URLs concurrently."""
    async with ContentProcessor(use_jina=use_jina, jina_api_key=jina_api_key) as processor:
        tasks = [processor.process_url(url, max_length) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert results to dictionary
        url_content = {}
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                url_content[url] = f"[Error: {str(result)}]"
            else:
                url_content[url] = result
        
        return url_content

def format_search_results(results: List[Dict[str, Any]], max_length: int = 2000) -> str:
    """Format search results into a readable string."""
    if not results:
        return "No search results found."
    
    formatted = []
    current_length = 0
    
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        url = result.get('url', 'No URL')
        snippet = result.get('snippet', 'No snippet available')
        
        result_str = f"[{i}] {title}\nURL: {url}\n{snippet}\n"
        
        # Check if adding this result would exceed max length
        if current_length + len(result_str) > max_length and i > 1:
            remaining = len(results) - i + 1
            if remaining > 0:
                formatted.append(f"... and {remaining} more results")
            break
        
        formatted.append(result_str)
        current_length += len(result_str)
    
    return "\n".join(formatted)

def extract_relevant_snippet(content: str, query: str, max_length: int = 300) -> str:
    """Extract relevant snippet from content based on query."""
    if not content or not query:
        return content[:max_length] if content else ""
    
    # Simple relevance extraction - find sentences containing query terms
    query_terms = query.lower().split()
    sentences = re.split(r'[.!?]+', content)
    
    relevant_sentences = []
    for sentence in sentences:
        if any(term in sentence.lower() for term in query_terms):
            relevant_sentences.append(sentence.strip())
    
    if relevant_sentences:
        snippet = '. '.join(relevant_sentences[:3])  # Take first 3 relevant sentences
        if len(snippet) > max_length:
            snippet = snippet[:max_length] + "..."
        return snippet
    
    # Fallback to beginning of content
    return content[:max_length] + ("..." if len(content) > max_length else "")
