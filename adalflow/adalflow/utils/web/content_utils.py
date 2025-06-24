"""Content processing utilities."""

import re
import html
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin, urlparse

def clean_html_content(html_content: str) -> str:
    """Clean HTML content by removing tags and formatting."""
    if not html_content:
        return ""
    
    try:
        from bs4 import BeautifulSoup
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        
    except ImportError:
        # Fallback without BeautifulSoup
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html_content)
        # Decode HTML entities
        text = html.unescape(text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_text_content(content: str, content_type: str = "html") -> str:
    """Extract text content based on content type."""
    if not content:
        return ""
    
    if content_type.lower() in ["html", "text/html"]:
        return clean_html_content(content)
    elif content_type.lower() in ["text", "text/plain"]:
        return content.strip()
    elif content_type.lower() in ["json", "application/json"]:
        try:
            import json
            data = json.loads(content)
            # Extract text from common JSON structures
            if isinstance(data, dict):
                text_parts = []
                for key, value in data.items():
                    if isinstance(value, str):
                        text_parts.append(f"{key}: {value}")
                return "\n".join(text_parts)
            elif isinstance(data, list):
                return "\n".join(str(item) for item in data)
            else:
                return str(data)
        except:
            return content
    else:
        return content

def truncate_content(
    content: str, 
    max_length: int = 1000, 
    preserve_sentences: bool = True
) -> str:
    """Truncate content while preserving readability."""
    if not content or len(content) <= max_length:
        return content
    
    if preserve_sentences:
        # Try to truncate at sentence boundaries
        sentences = re.split(r'[.!?]+', content)
        truncated = ""
        
        for sentence in sentences:
            if len(truncated) + len(sentence) + 1 <= max_length:
                truncated += sentence + "."
            else:
                break
        
        if truncated:
            return truncated.strip()
    
    # Fallback to simple truncation
    return content[:max_length] + "..." if len(content) > max_length else content

def extract_links(html_content: str, base_url: Optional[str] = None) -> List[Dict[str, str]]:
    """Extract links from HTML content."""
    links = []
    
    try:
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)
            
            # Convert relative URLs to absolute
            if base_url and not href.startswith(('http://', 'https://')):
                href = urljoin(base_url, href)
            
            if href and text:
                links.append({
                    'url': href,
                    'text': text,
                    'title': link.get('title', '')
                })
    
    except ImportError:
        # Fallback regex-based extraction
        link_pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>'
        matches = re.findall(link_pattern, html_content, re.IGNORECASE)
        
        for href, text in matches:
            if base_url and not href.startswith(('http://', 'https://')):
                href = urljoin(base_url, href)
            
            links.append({
                'url': href,
                'text': text.strip(),
                'title': ''
            })
    
    return links

def detect_content_language(content: str) -> str:
    """Detect the language of content (simple heuristic-based approach)."""
    if not content:
        return "unknown"
    
    # Simple language detection based on common words
    content_lower = content.lower()
    
    # English indicators
    english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    english_count = sum(1 for word in english_words if word in content_lower)
    
    # Spanish indicators
    spanish_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo']
    spanish_count = sum(1 for word in spanish_words if word in content_lower)
    
    # French indicators
    french_words = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour']
    french_count = sum(1 for word in french_words if word in content_lower)
    
    # German indicators
    german_words = ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf']
    german_count = sum(1 for word in german_words if word in content_lower)
    
    # Determine language based on highest count
    language_scores = {
        'english': english_count,
        'spanish': spanish_count,
        'french': french_count,
        'german': german_count
    }
    
    detected_language = max(language_scores, key=language_scores.get)
    
    # Return 'english' as default if no clear winner or if score is too low
    if language_scores[detected_language] < 3:
        return 'english'
    
    return detected_language

def extract_metadata(html_content: str) -> Dict[str, Any]:
    """Extract metadata from HTML content."""
    metadata = {
        'title': '',
        'description': '',
        'keywords': '',
        'author': '',
        'language': '',
        'charset': ''
    }
    
    try:
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text(strip=True)
        
        # Extract meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name', '').lower()
            property_name = meta.get('property', '').lower()
            content = meta.get('content', '')
            
            if name == 'description' or property_name == 'og:description':
                metadata['description'] = content
            elif name == 'keywords':
                metadata['keywords'] = content
            elif name == 'author':
                metadata['author'] = content
            elif name == 'language' or meta.get('http-equiv', '').lower() == 'content-language':
                metadata['language'] = content
            elif meta.get('charset'):
                metadata['charset'] = meta.get('charset')
    
    except ImportError:
        # Fallback regex-based extraction
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        
        # Extract description
        desc_match = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']', html_content, re.IGNORECASE)
        if desc_match:
            metadata['description'] = desc_match.group(1)
    
    return metadata

def summarize_content(content: str, max_sentences: int = 3) -> str:
    """Create a summary of content by extracting key sentences."""
    if not content:
        return ""
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    if len(sentences) <= max_sentences:
        return '. '.join(sentences) + '.'
    
    # Simple extractive summarization
    # Score sentences based on length and position
    scored_sentences = []
    
    for i, sentence in enumerate(sentences):
        # Prefer sentences that are not too short or too long
        length_score = 1.0 if 20 <= len(sentence) <= 200 else 0.5
        
        # Prefer sentences near the beginning
        position_score = max(0.5, 1.0 - (i / len(sentences)))
        
        total_score = length_score * position_score
        scored_sentences.append((sentence, total_score))
    
    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [s[0] for s in scored_sentences[:max_sentences]]
    
    # Reorder by original position in text
    summary_sentences = []
    for sentence in sentences:
        if sentence in top_sentences:
            summary_sentences.append(sentence)
    
    return '. '.join(summary_sentences) + '.'

def clean_and_normalize_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-printable characters except newlines
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    
    # Normalize quotes
    text = re.sub(r'[""''`]', '"', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    
    # Clean up multiple spaces again
    text = re.sub(r' +', ' ', text)
    
    return text.strip()
