"""Caching utilities for web search operations."""

import json
import hashlib
import time
from typing import Dict, Any, Optional, Union
from pathlib import Path
import pickle
import logging

log = logging.getLogger(__name__)

def create_cache_key(query: str, **kwargs) -> str:
    """Create a cache key from query and parameters."""
    # Normalize query
    query_normalized = query.lower().strip()
    
    # Create key from query and parameters
    key_data = {
        'query': query_normalized,
        **kwargs
    }
    
    # Create hash
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()

class SearchCache:
    """Cache for search results to avoid duplicate API calls."""
    
    def __init__(
        self, 
        cache_file: Optional[str] = None,
        max_age_seconds: int = 3600,  # 1 hour default
        max_entries: int = 1000
    ):
        self.cache_file = Path(cache_file) if cache_file else None
        self.max_age_seconds = max_age_seconds
        self.max_entries = max_entries
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        # Load existing cache if file exists
        if self.cache_file and self.cache_file.exists():
            self.load_cache()
    
    def get(self, query: str, **kwargs) -> Optional[Any]:
        """Get cached search results."""
        cache_key = create_cache_key(query, **kwargs)
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            
            # Check if entry is still valid
            if time.time() - entry['timestamp'] < self.max_age_seconds:
                log.debug(f"Cache hit for query: {query}")
                return entry['data']
            else:
                # Remove expired entry
                del self.cache[cache_key]
                log.debug(f"Cache entry expired for query: {query}")
        
        log.debug(f"Cache miss for query: {query}")
        return None
    
    def set(self, query: str, data: Any, **kwargs) -> None:
        """Cache search results."""
        cache_key = create_cache_key(query, **kwargs)
        
        # Enforce max entries limit
        if len(self.cache) >= self.max_entries:
            # Remove oldest entries
            oldest_keys = sorted(
                self.cache.keys(), 
                key=lambda k: self.cache[k]['timestamp']
            )
            
            # Remove oldest 10% of entries
            num_to_remove = max(1, len(oldest_keys) // 10)
            for key in oldest_keys[:num_to_remove]:
                del self.cache[key]
        
        # Add new entry
        self.cache[cache_key] = {
            'data': data,
            'timestamp': time.time(),
            'query': query,
            'params': kwargs
        }
        
        log.debug(f"Cached results for query: {query}")
        
        # Save to file if configured
        if self.cache_file:
            self.save_cache()
    
    def has(self, query: str, **kwargs) -> bool:
        """Check if query is cached and still valid."""
        return self.get(query, **kwargs) is not None
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        if self.cache_file and self.cache_file.exists():
            self.cache_file.unlink()
        log.info("Cache cleared")
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time - entry['timestamp'] >= self.max_age_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys and self.cache_file:
            self.save_cache()
        
        log.info(f"Removed {len(expired_keys)} expired cache entries")
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        valid_entries = 0
        expired_entries = 0
        
        for entry in self.cache.values():
            if current_time - entry['timestamp'] < self.max_age_seconds:
                valid_entries += 1
            else:
                expired_entries += 1
        
        return {
            'total_entries': len(self.cache),
            'valid_entries': valid_entries,
            'expired_entries': expired_entries,
            'max_entries': self.max_entries,
            'max_age_seconds': self.max_age_seconds
        }
    
    def save_cache(self) -> None:
        """Save cache to file."""
        if not self.cache_file:
            return
        
        try:
            # Ensure directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            
            log.debug(f"Cache saved to {self.cache_file}")
            
        except Exception as e:
            log.error(f"Error saving cache: {str(e)}")
    
    def load_cache(self) -> None:
        """Load cache from file."""
        if not self.cache_file or not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            
            log.debug(f"Cache loaded from {self.cache_file}")
            
            # Clean up expired entries
            self.cleanup_expired()
            
        except Exception as e:
            log.error(f"Error loading cache: {str(e)}")
            self.cache = {}

class URLCache:
    """Cache for URL content to avoid re-fetching same pages."""
    
    def __init__(
        self, 
        cache_file: Optional[str] = None,
        max_age_seconds: int = 7200,  # 2 hours default
        max_entries: int = 500,
        max_content_size: int = 100000  # 100KB per entry
    ):
        self.cache_file = Path(cache_file) if cache_file else None
        self.max_age_seconds = max_age_seconds
        self.max_entries = max_entries
        self.max_content_size = max_content_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        # Load existing cache if file exists
        if self.cache_file and self.cache_file.exists():
            self.load_cache()
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for consistent caching."""
        # Remove fragment and query parameters for caching key
        from urllib.parse import urlparse
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return normalized.lower()
    
    def get(self, url: str) -> Optional[str]:
        """Get cached URL content."""
        cache_key = self._normalize_url(url)
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            
            # Check if entry is still valid
            if time.time() - entry['timestamp'] < self.max_age_seconds:
                log.debug(f"URL cache hit: {url}")
                return entry['content']
            else:
                # Remove expired entry
                del self.cache[cache_key]
                log.debug(f"URL cache entry expired: {url}")
        
        log.debug(f"URL cache miss: {url}")
        return None
    
    def set(self, url: str, content: str) -> None:
        """Cache URL content."""
        cache_key = self._normalize_url(url)
        
        # Skip if content is too large
        if len(content) > self.max_content_size:
            log.debug(f"Content too large to cache: {url} ({len(content)} bytes)")
            return
        
        # Enforce max entries limit
        if len(self.cache) >= self.max_entries:
            # Remove oldest entries
            oldest_keys = sorted(
                self.cache.keys(), 
                key=lambda k: self.cache[k]['timestamp']
            )
            
            # Remove oldest 10% of entries
            num_to_remove = max(1, len(oldest_keys) // 10)
            for key in oldest_keys[:num_to_remove]:
                del self.cache[key]
        
        # Add new entry
        self.cache[cache_key] = {
            'content': content,
            'timestamp': time.time(),
            'url': url,
            'size': len(content)
        }
        
        log.debug(f"Cached URL content: {url} ({len(content)} bytes)")
        
        # Save to file if configured
        if self.cache_file:
            self.save_cache()
    
    def has(self, url: str) -> bool:
        """Check if URL is cached and still valid."""
        return self.get(url) is not None
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        if self.cache_file and self.cache_file.exists():
            self.cache_file.unlink()
        log.info("URL cache cleared")
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time - entry['timestamp'] >= self.max_age_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys and self.cache_file:
            self.save_cache()
        
        log.info(f"Removed {len(expired_keys)} expired URL cache entries")
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        total_size = 0
        valid_entries = 0
        expired_entries = 0
        
        for entry in self.cache.values():
            total_size += entry['size']
            if current_time - entry['timestamp'] < self.max_age_seconds:
                valid_entries += 1
            else:
                expired_entries += 1
        
        return {
            'total_entries': len(self.cache),
            'valid_entries': valid_entries,
            'expired_entries': expired_entries,
            'total_size_bytes': total_size,
            'average_size_bytes': total_size // len(self.cache) if self.cache else 0,
            'max_entries': self.max_entries,
            'max_age_seconds': self.max_age_seconds
        }
    
    def save_cache(self) -> None:
        """Save cache to file."""
        if not self.cache_file:
            return
        
        try:
            # Ensure directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            
            log.debug(f"URL cache saved to {self.cache_file}")
            
        except Exception as e:
            log.error(f"Error saving URL cache: {str(e)}")
    
    def load_cache(self) -> None:
        """Load cache from file."""
        if not self.cache_file or not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            
            log.debug(f"URL cache loaded from {self.cache_file}")
            
            # Clean up expired entries
            self.cleanup_expired()
            
        except Exception as e:
            log.error(f"Error loading URL cache: {str(e)}")
            self.cache = {}
