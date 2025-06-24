"""Web utilities for AdalFlow."""

from .search_utils import (
    validate_search_query,
    clean_search_query,
    extract_search_tokens,
    format_search_results,
)

from .content_utils import (
    clean_html_content,
    extract_text_content,
    truncate_content,
    detect_content_language,
)

from .cache_utils import (
    SearchCache,
    URLCache,
    create_cache_key,
)

__all__ = [
    "validate_search_query",
    "clean_search_query", 
    "extract_search_tokens",
    "format_search_results",
    "clean_html_content",
    "extract_text_content",
    "truncate_content",
    "detect_content_language",
    "SearchCache",
    "URLCache",
    "create_cache_key",
]
