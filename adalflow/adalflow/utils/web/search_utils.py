"""Search utility functions."""

import re
from typing import List, Dict, Any, Optional, Tuple

# Invalid search queries to filter out
INVALID_SEARCH_QUERIES = [
    "and end with",
    "search query",
    "query",
    "your query here",
    "your query",
    "your search query",
    "search for",
    "look up",
    "find information about",
    "search term",
    "keyword",
    "keywords",
]

# Search token patterns
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

def validate_search_query(query: str) -> bool:
    """Validate if a search query is valid and useful."""
    if not query or not isinstance(query, str):
        return False
    
    # Check minimum length
    if len(query.strip()) <= 3:
        return False
    
    # Check if query is in invalid list
    query_lower = query.lower().strip()
    if query_lower in INVALID_SEARCH_QUERIES:
        return False
    
    # Check if query contains search tokens (shouldn't be in actual query)
    if BEGIN_SEARCH_QUERY in query or END_SEARCH_QUERY in query:
        return False
    
    # Check if query is too generic
    generic_patterns = [
        r'^(what|how|when|where|why|who)\s*(is|are|was|were|do|does|did)?\s*$',
        r'^(search|find|look)\s*(for|up)?\s*$',
        r'^(information|info|details|data)\s*(about|on)?\s*$',
    ]
    
    for pattern in generic_patterns:
        if re.match(pattern, query_lower):
            return False
    
    return True

def clean_search_query(query: str) -> str:
    """Clean and normalize a search query."""
    if not query:
        return ""
    
    # Remove extra whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    # Remove special characters that might interfere with search
    query = re.sub(r'[<>|]', '', query)
    
    # Remove search tokens if present
    query = query.replace(BEGIN_SEARCH_QUERY, '')
    query = query.replace(END_SEARCH_QUERY, '')
    
    # Remove quotes if they wrap the entire query
    if query.startswith('"') and query.endswith('"'):
        query = query[1:-1]
    
    return query.strip()

def extract_search_tokens(text: str) -> List[Dict[str, Any]]:
    """Extract search queries and results from text with special tokens."""
    tokens = []
    
    # Pattern for search queries
    query_pattern = rf"{re.escape(BEGIN_SEARCH_QUERY)}(.*?){re.escape(END_SEARCH_QUERY)}"
    query_matches = re.finditer(query_pattern, text, re.DOTALL)
    
    for match in query_matches:
        query = match.group(1).strip()
        tokens.append({
            'type': 'search_query',
            'content': query,
            'start': match.start(),
            'end': match.end(),
            'valid': validate_search_query(query)
        })
    
    # Pattern for search results
    result_pattern = rf"{re.escape(BEGIN_SEARCH_RESULT)}(.*?){re.escape(END_SEARCH_RESULT)}"
    result_matches = re.finditer(result_pattern, text, re.DOTALL)
    
    for match in result_matches:
        result = match.group(1).strip()
        tokens.append({
            'type': 'search_result',
            'content': result,
            'start': match.start(),
            'end': match.end()
        })
    
    # Sort by position in text
    tokens.sort(key=lambda x: x['start'])
    
    return tokens

def format_search_results(
    results: List[Dict[str, Any]], 
    max_results: Optional[int] = None,
    max_length: Optional[int] = None
) -> str:
    """Format search results into a readable string."""
    if not results:
        return "No search results found."
    
    # Limit number of results
    if max_results:
        results = results[:max_results]
    
    formatted_results = []
    current_length = 0
    
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        url = result.get('url', result.get('link', 'No URL'))
        snippet = result.get('snippet', result.get('description', 'No description available'))
        
        # Format individual result
        result_text = f"[{i}] {title}\nURL: {url}\n{snippet}\n"
        
        # Check length limit
        if max_length and current_length + len(result_text) > max_length:
            remaining = len(results) - i + 1
            if remaining > 0:
                formatted_results.append(f"... and {remaining} more results")
            break
        
        formatted_results.append(result_text)
        current_length += len(result_text)
    
    return "\n".join(formatted_results)

def extract_search_intent(reasoning_text: str, search_query: str) -> str:
    """Extract search intent from reasoning context."""
    if not reasoning_text or not search_query:
        return "General information search"
    
    # Look for intent indicators in the reasoning
    intent_patterns = [
        r"(?:need to|want to|looking for|searching for|trying to find)\s+(.+?)(?:\.|$)",
        r"(?:information about|details on|data about)\s+(.+?)(?:\.|$)",
        r"(?:how|what|when|where|why|who)\s+(.+?)(?:\?|$)",
    ]
    
    reasoning_lower = reasoning_text.lower()
    
    for pattern in intent_patterns:
        matches = re.findall(pattern, reasoning_lower)
        if matches:
            intent = matches[-1].strip()  # Take the last/most recent intent
            if len(intent) > 10:  # Filter out too short intents
                return intent
    
    # Fallback to using the search query as intent
    return f"Find information about: {search_query}"

def calculate_search_relevance(query: str, title: str, snippet: str) -> float:
    """Calculate relevance score between query and search result."""
    if not query or (not title and not snippet):
        return 0.0
    
    query_terms = set(query.lower().split())
    
    # Calculate title relevance (weighted more heavily)
    title_terms = set(title.lower().split()) if title else set()
    title_overlap = len(query_terms.intersection(title_terms))
    title_score = title_overlap / len(query_terms) if query_terms else 0
    
    # Calculate snippet relevance
    snippet_terms = set(snippet.lower().split()) if snippet else set()
    snippet_overlap = len(query_terms.intersection(snippet_terms))
    snippet_score = snippet_overlap / len(query_terms) if query_terms else 0
    
    # Weighted combination (title more important)
    relevance_score = (title_score * 0.7) + (snippet_score * 0.3)
    
    return min(relevance_score, 1.0)  # Cap at 1.0

def rank_search_results(
    results: List[Dict[str, Any]], 
    query: str,
    max_results: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Rank search results by relevance to query."""
    if not results:
        return []
    
    # Calculate relevance scores
    scored_results = []
    for result in results:
        title = result.get('title', '')
        snippet = result.get('snippet', result.get('description', ''))
        relevance = calculate_search_relevance(query, title, snippet)
        
        scored_result = result.copy()
        scored_result['relevance_score'] = relevance
        scored_results.append(scored_result)
    
    # Sort by relevance score (descending)
    scored_results.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Limit results if specified
    if max_results:
        scored_results = scored_results[:max_results]
    
    return scored_results

def merge_search_results(
    results_list: List[List[Dict[str, Any]]], 
    query: str,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """Merge multiple search result lists and deduplicate."""
    if not results_list:
        return []
    
    # Flatten all results
    all_results = []
    seen_urls = set()
    
    for results in results_list:
        for result in results:
            url = result.get('url', result.get('link', ''))
            
            # Skip duplicates based on URL
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(result)
    
    # Rank merged results
    ranked_results = rank_search_results(all_results, query, max_results)
    
    return ranked_results
