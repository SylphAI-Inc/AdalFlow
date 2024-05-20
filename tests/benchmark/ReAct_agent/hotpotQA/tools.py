import requests
from bs4 import BeautifulSoup

"""
https://arxiv.org/abs/2210.03629, published in Mar, 2023

Apply the similar code for wikipedia search from the Paper (open-source).
"""

# langchain simply uses requests.post and gets the response.json()
def search(entity: str) -> str:
    """
    Searches for a given entity on Wikipedia and retrieves the first few paragraphs of the page

    Args:
        entity (str): name string of the entity

    Returns:
        str: The entity related information on Wikipedia, or similar search results for the entity if the exact entity page isn't found.
    """
    # Format the entity for URL encoding
    entity_formatted = entity.replace(" ", "+")
    url = f"https://en.wikipedia.org/w/index.php?search={entity_formatted}"
    
    # Fetch the page
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Check if the exact page was found or suggest similar items
    # when <div class=mw-search-result-heading> is detected, it means the entity is not found on wikipedia
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    
    if result_divs: # this means the searched entity is not in wikipedia, and wikipedia shows a list of similar entities
        # Similar results found, not the exact page
        similar_titles = [div.a.get_text() for div in result_divs]
        return f"Could not find exact page for '{entity}'. Similar topics: {similar_titles[:5]}" # return the top 5 similar titles
    else:
        # p is each paragraph in the searched wiki page; ul is the additional content
        # Extract paragraphs content from the page
        paragraphs = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        
        # TODO: Recursive search, if find any concept that needs more search then call search again
        # if any("may refer to:" in p for p in paragraphs):
        #     search(entity)
        
        # TODO: clean the paragraphs and return the searched content
        
        if paragraphs:
            return ' '.join(paragraphs[:5])  # Return the first 5 paragraphs
        else:
            return "No content found on this page."


def lookup(text: str, keyword: str) -> str:
    """Looks for sentences containing a specific keyword in the provided text

    Args:
        text (str)
        keyword (str)

    Returns:
        str: sententces containing the keyword
    """
    sentences = text.split('.')
    matching_sentences = [sentence.strip() + '.' for sentence in sentences if keyword.lower() in sentence.lower()]
    if not matching_sentences:
        return "No sentences found with the keyword."
    else:
        return ' '.join(matching_sentences)  # Join all matching sentences into a single string
