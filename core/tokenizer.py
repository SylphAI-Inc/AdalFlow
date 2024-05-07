import tiktoken

from typing import List


##############################################
# Helper modules for RAG
##############################################
class Tokenizer:
    def __init__(self, name: str = "cl100k_base"):
        self.name = name
        self.tokenizer = tiktoken.get_encoding(name)

    def encode(self, text: str) -> List[str]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[str]) -> str:
        return self.tokenizer.decode(tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))
