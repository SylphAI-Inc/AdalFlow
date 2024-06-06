import tiktoken
from typing import List

from lightrag.core.component import Component


class Tokenizer(Component):
    """
    Tokenizer component that wraps around the tokenizer from tiktoken.
    __call__ is the same as forward/encode, so that we can use it in Sequential
    Additonally, you can can also use encode and decode methods.
    """

    def __init__(self, name: str = "cl100k_base"):
        super().__init__()
        self.name = name
        self.tokenizer = tiktoken.get_encoding(name)

    # call is the same as forward/encode, so that we can use it in Sequential
    def __call__(self, input: str) -> List[str]:
        return self.encode(input)

    def encode(self, text: str) -> List[str]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[str]) -> str:
        return self.tokenizer.decode(tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))
