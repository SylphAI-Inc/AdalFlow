"""
Here is for us to prepare the documents for retrieve the context.
LlamaIndex having DocumentStore. We just want you wrap your data here and define a retrive method. 
It highly depends on the product environment and can go beyond the scope of this library.

But these are shared:
* DocumentStore
* Chunk Document -> VectorStore
* Embed chunk
* Openup the db for context retrieval
"""

# TODO: (1) TextSplitters
from copy import deepcopy
from typing import List, Literal

from more_itertools import windowed

from core.component import Component
from core.data_classes import Document

# TODO: convert this to function

DocumentSplitterInputType = List[Document]
DocumentSplitterOutputType = List[Document]


class DocumentSplitter(Component):
    r"""
    Splits a list of text documents into a list of text documents with shorter texts.

    Output: List[Document]

    Splitting documents with long texts is a common preprocessing step during indexing.
    This allows Embedders to create significant semantic representations
    and avoids exceeding the maximum context length of language models.
    """

    def __init__(
        self,
        split_by: Literal["word", "sentence", "page", "passage"] = "word",
        split_length: int = 200,
        split_overlap: int = 0,
    ):
        """
        :param split_by: The unit by which the document should be split. Choose from "word" for splitting by " ",
            "sentence" for splitting by ".", "page" for splitting by "\\f" or "passage" for splitting by "\\n\\n".
        :param split_length: The maximum number of units in each split.
        :param split_overlap: The number of units that each split should overlap.
        """
        super().__init__()

        self.split_by = split_by
        if split_by not in ["word", "sentence", "page", "passage"]:
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
        """
        Splits a text into a list of shorter texts.

        :param text: The text to split.

        :returns: A list of shorter texts.
        """
        units = self._split_into_units(text, self.split_by)
        return self._concatenate_units(units, self.split_length, self.split_overlap)

    def call(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents by the unit expressed in `split_by`, with a length of `split_length`
        and an overlap of `split_overlap`.

        :param documents: The documents to split.

        :returns: A dictionary with the following key:
            - `documents`: List of documents with the split texts. A metadata field "source_id" is added to each
            document to keep track of the original document that was split. Other metadata are copied from the original
            document.

        :raises TypeError: if the input is not a list of Documents.
        :raises ValueError: if the content of a document is None.
        """

        print(f"start to split documents")

        if not isinstance(documents, list) or (
            documents and not isinstance(documents[0], Document)
        ):
            raise TypeError("DocumentSplitter expects a List of Documents as input.")

        split_docs = []
        for doc in documents:
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
        print(f"splitted_doc: {split_docs}")
        return split_docs

    def _split_into_units(
        self, text: str, split_by: Literal["word", "sentence", "passage", "page"]
    ) -> List[str]:
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
        """
        Concatenates the elements into parts of split_length units.
        """
        text_splits = []
        segments = windowed(elements, n=split_length, step=split_length - split_overlap)
        for seg in segments:
            current_units = [unit for unit in seg if unit is not None]
            txt = "".join(current_units)
            if len(txt) > 0:
                text_splits.append(txt)
        return text_splits

    def extra_repr(self) -> str:
        s = f"split_by={self.split_by}, split_length={self.split_length}, split_overlap={self.split_overlap}"
        return s
