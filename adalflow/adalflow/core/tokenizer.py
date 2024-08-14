"""
Tokenizer from tiktoken.
"""

import tiktoken
from typing import List

# from adalflow.core.component import BaseComponent


class Tokenizer:
    __doc__ = r"""
    Tokenizer component that wraps around the tokenizer from tiktoken.
    __call__ is the same as forward/encode, so that we can use it in Sequential
    Additonally, you can can also use encode and decode methods.

    Args:
        name (str, optional): The name of the tokenizer. Defaults to "cl100k_base". You can find more information
        at the tiktoken documentation.
    """

    def __init__(self, name: str = "cl100k_base", remove_stop_words: bool = False):
        super().__init__()
        self.name = name
        self.tokenizer = tiktoken.get_encoding(name)
        self.stop_words = (
            set(["and", "the", "is", "in", "at", "of", "a", "an"])
            if remove_stop_words
            else set()
        )

    # call is the same as forward/encode, so that we can use it in Sequential
    def __call__(self, input: str) -> List[str]:
        return self.encode(input)

    def preprocess(self, text: str) -> List[str]:
        # Lowercase the text
        words = text.lower().split()
        return words

    def encode(self, text: str) -> List[int]:
        r"""Encodes the input text/word into token IDs."""
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[str]) -> str:
        r"""Decodes the input tokens into text."""
        return self.tokenizer.decode(tokens)

    def count_tokens(self, text: str) -> int:
        r"""Counts the number of tokens in the input text."""
        return len(self.encode(text))

    def get_string_tokens(self, text: str) -> List[str]:
        r"""Returns the string tokens from the input text."""
        token_ids = self.encode(text)
        return [self.tokenizer.decode([token_id]) for token_id in token_ids]
