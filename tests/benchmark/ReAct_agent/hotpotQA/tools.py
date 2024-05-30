import requests
from bs4 import BeautifulSoup
import re
import string

"""
https://arxiv.org/abs/2210.03629, published in Mar, 2023

Apply the similar code for wikipedia search from the Paper (open-source).
"""

# copy code from the paper
def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

# normalization copied from the paper's code
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
  
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# langchain simply uses requests.post and gets the response.json()
def search(entity: str) -> str:
    """
    searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
    """
    # Format the entity for URL encoding
    entity_formatted = entity.replace(" ", "+")
    url = f"https://en.wikipedia.org/w/index.php?search={entity_formatted}"
    
    # Fetch the page
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Check if the exact page was found or suggest similar items
    # when <div class=mw-search-result-heading> is detected, it means the entity page is not found on wikipedia
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    
    if result_divs: # this means the searched entity page is not in wikipedia, wikipedia will show a list of similar entities
        # get Similar results
        similar_titles = [div.a.get_text() for div in result_divs]
        return f"Could not find exact page for '{entity}'. Similar topics: {similar_titles[:5]}" # return the top 5 similar titles
    else:
        # the paper uses page to represent content in <p>
        # Extract xontent
        page_list = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        # TODO: Recursive search, if find any concept that needs more search then call search again
        # if any("may refer to:" in p for p in page_list):
        #     search(entity)

        # restructure & clean the page content following the paper's logic
        page = ''
        for p in page_list:
            if len(p.split(" ")) > 2:
                page += clean_str(p)
                if not p.endswith("\n"):
                    page += "\n"
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        # return the first 5 sentences
        if sentences:
            return ' '.join(sentences[:5]) if len(sentences)>=5 else ' '.join(sentences)
        else:
            return "No content found on this page."
        
        # TODO: clean the paragraphs and return the searched content


def lookup(text: str, keyword: str) -> str:
    """
        returns the sentences containing keyword in the current passage.
    """
    sentences = text.split('.')
    matching_sentences = [sentence.strip() + '.' for sentence in sentences if keyword.lower() in sentence.lower()]
    if not matching_sentences:
        return "No sentences found with the keyword."
    else:
        return ' '.join(matching_sentences)  # Join all matching sentences into a single string
