"""Text splitter to split long text into smaller chunks to fit into the token limits of embedding and LLM models."""

# TODO: JSON/HTML Splitter
from copy import deepcopy
from typing import List, Literal
from tqdm import tqdm

from more_itertools import windowed

from lightrag.core.component import Component
from lightrag.core.types import Document
from lightrag.core.tokenizer import Tokenizer


DocumentSplitterInputType = List[Document]
DocumentSplitterOutputType = List[Document]


def split_text_by_token_fn(x: str, tokenizer: Tokenizer = Tokenizer()) -> List[str]:
    x = x.lower()
    return tokenizer.get_string_tokens(x)


DocumentSplitterInputType = List[Document]
DocumentSplitterOutputType = List[Document]


class DocumentSplitter(Component):
    __doc__ = r"""
    Splits a list of text documents into a list of text documents with shorter texts.

    Output: List[Document]

    Splitting documents with long texts is a common preprocessing step for LLM applications.
    The splitted documents are easier to fit into the token limits of language models, both Embedders and Generators,
    and to ensure the retrieved context can be more relevant than the large text itself.

    Args:
        split_by (str): The unit by which the document should be split. Choose from "word" for splitting by " ",
            "sentence" for splitting by ".", "page" for splitting by "\\f" or "passage" for splitting by "\\n\\n".
        split_length (int): The maximum number of units in each split. It can be number of works, sentences, pages or passages.
        split_overlap (int): The number of units that each split should overlap.

    Example:

    .. code-block:: python

        from lightrag.core.document_splitter import DocumentSplitter
        from lightrag.core.types import Document

        doc1 = Document(text="This is a test document. It is a long document.")
        doc2 = Document(text="This is another test document. It is also a long document.")
        splitter = DocumentSplitter(split_by="token", split_length=4, split_overlap=1)
        print(splitter)
        splitted_docs = splitter([doc1, doc2])
        print(splitted_docs)
    """

    def __init__(
        self,
        split_by: Literal["word", "token", "sentence", "page", "passage"] = "word",
        split_length: int = 200,
        split_overlap: int = 0,
    ):
        super().__init__(
            split_by=split_by, split_length=split_length, split_overlap=split_overlap
        )

        self.split_by = split_by
        if split_by not in ["word", "sentence", "page", "passage", "token"]:
            raise ValueError(
                "split_by must be one of 'word', 'sentence', 'page' or 'passage'."
            )
        if split_length <= 0:
            raise ValueError("split_length must be greater than 0.")
        self.split_length = split_length
        if split_overlap < 0:
            raise ValueError("split_overlap must be greater than or equal to 0.")
        self.split_overlap = split_overlap

    def split_text(self, text: str) -> List[str]:
        r"""Splits a text into a list of shorter texts."""
        units = self._split_into_units(text, self.split_by)
        return self._concatenate_units(units, self.split_length, self.split_overlap)

    def call(self, documents: List[Document]) -> List[Document]:
        if not isinstance(documents, list) or (
            documents and not isinstance(documents[0], Document)
        ):
            raise TypeError("DocumentSplitter expects a List of Documents as input.")

        split_docs: List[Document] = []
        for doc in tqdm(documents, desc="Splitting documents"):
            if doc.text is None:
                raise ValueError(
                    f"DocumentSplitter only works with text documents but document.content for document ID {doc.id} is None."
                )
            text_splits = self.split_text(doc.text)
            meta_data = deepcopy(doc.meta_data)
            split_docs += [
                Document(
                    text=txt,
                    meta_data=meta_data,
                    parent_doc_id=f"{doc.id}",
                    order=i,
                    vector=[],
                )
                for i, txt in enumerate(text_splits)
            ]
        return split_docs

    def _split_into_units(
        self,
        text: str,
        split_by: Literal["word", "sentence", "passage", "page", "token"],
    ) -> List[str]:
        if split_by == "token":
            units = split_text_by_token_fn(x=text)
            print(units)
        else:  # text splitter
            if split_by == "page":
                split_at = "\f"
            elif split_by == "passage":
                split_at = "\n\n"
            elif split_by == "sentence":
                split_at = "."
            elif split_by == "word":
                split_at = " "
            else:
                raise NotImplementedError(
                    "DocumentSplitter only supports 'word', 'sentence', 'page' or 'passage' split_by options."
                )
            units = text.split(split_at)
            # Add the delimiter back to all units except the last one
            for i in range(len(units) - 1):
                units[i] += split_at
        return units

    def _concatenate_units(
        self, elements: List[str], split_length: int, split_overlap: int
    ) -> List[str]:
        """Concatenates the elements into parts of split_length units."""
        text_splits = []
        segments = windowed(elements, n=split_length, step=split_length - split_overlap)
        for seg in segments:
            current_units = [unit for unit in seg if unit is not None]
            txt = "".join(current_units)
            if len(txt) > 0:
                text_splits.append(txt)
        return text_splits

    def _extra_repr(self) -> str:
        s = f"split_by={self.split_by}, split_length={self.split_length}, split_overlap={self.split_overlap}"
        return s
    
if __name__ == "__main__":
    from lightrag.core.document_splitter import DocumentSplitter
    from lightrag.core.types import Document

    doc1 = Document(text="This is a simple test to check splitting.")
    # doc2 = Document(text="This is another test document. It is also a long document.")
    splitter = DocumentSplitter(split_by="word", split_length=5, split_overlap=2)
    # print(splitter)
    splitted_docs = splitter([doc1])
    # print(splitted_docs)
    for doc in splitted_docs:
        print(doc.text)