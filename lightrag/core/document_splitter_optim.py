"""
Document Splitter

Document Splitter is for us to prepare the documents for retrieve the context.
LlamaIndex having DocumentStore. We just want you wrap your data here and define a retrieve method. 
It highly depends on the product environment and can go beyond the scope of this library.

But these are shared:
* DocumentStore
* Chunk Document -> VectorStore
* Embed chunk
* Openup the db for context retrieval
"""

from copy import deepcopy
from typing import List, Literal

from lightrag.core.component import Component
from lightrag.core.types import Document
from tqdm import tqdm

# TODO: convert this to function

DocumentSplitterInputType = List[Document]
DocumentSplitterOutputType = List[Document]

# customizable seperators
separators = {"page": "\f", "passage": "\n\n", "word": " ", "sentence": "."}

DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 20

class TextSplitter(Component):
    """
    Text splitter for processing and chunking documents.
    
    This splitter is versatile, supporting plain text splitting.
    Developers can customize the separators map to handle different formats effectively.
    Example: separators = {"question": "?"}.
    More splitters such as PDF/JSON/HTML Splitter can be built on TextSplitter.
    # TODO: PDF/JSON/HTML Splitter
    
    .. note::
        Our splitting is based on Python str.split(). The specified seperator will be added back to the end of each split.
        E.g. "hello world!" split by "word" will keeped as ["hello ", "world!"]
        If developers need to split PDFs, `PyPDF2` or `PDFMiner` can be utilized to extract the txt content first.
    
    Example:
        .. code-block:: python
        
            from lightrag.core.document_splitter_optim import TextSplitter
            from lightrag.core.types import Document

            # configure the splitter setting
            text_splitter_settings = {
                    "split_by": "word",
                    "chunk_size": 20,
                    "chunk_overlap": 2,
                    }

            # set up the document splitter
            text_splitter = TextSplitter(
                split_by=text_splitter_settings["split_by"],
                chunk_size=text_splitter_settings["chunk_size"],
                chunk_overlap=text_splitter_settings["chunk_overlap"],
                )

            doc1 = Document(
                meta_data={"title": "Luna's Profile"},
                text="lots of more nonsense text." * 2
                + "Luna is a domestic shorthair." 
                + "lots of nonsense text." * 3,
                id="doc1",
                )
            doc2 = Document(
                meta_data={"title": "Luna's Hobbies"},
                text="lots of more nonsense text." * 2
                + "Luna loves to eat lickable treats."
                + "lots of more nonsense text." * 2
                + "Luna loves to play cat wand." 
                + "lots of more nonsense text." * 2
                + "Luna likes to sleep all the afternoon",
                id="doc2",
            )
            documents = [doc1, doc2]

            splitted_docs = (text_splitter.call(documents=documents))

            for doc in splitted_docs:
                print("*" * 50)
                print(doc)
                print("*" * 50)
    """
    def __init__(
        self,
        split_by: Literal["word", "sentence", "page", "passage"] = "word",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """
        Initializes the TextSplitter with the specified parameters for text splitting.

        Args:
            split_by (str): The specific criterion to use for splitting the text. 
                            Valid options are 'word' to split by ' ', 'sentence' to split by '.', 
                            'page' to split by '\\f', 'passage' to split by '\\n\\n'.
            chunk_size (int): The size of chunks to generate after splitting. Must be greater than 0.
            chunk_overlap (int): The number of characters of overlap between chunks. Must be non-negative
                                and less than chunk_size.
        Raises:
            ValueError: If the provided split_by is not supported, chunk_size is not greater than 0,
                        or chunk_overlap is not valid as per the given conditions.
        """
        super().__init__()

        # variable value checks
        self.split_by = split_by
        if split_by not in separators:
            options = ", ".join(f"'{key}'" for key in separators.keys())
            raise ValueError(f"Invalid options for split_by. You must select from {options}.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap can't be larger than chunk_size. Received chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}"
            )
            
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be greater than 0. Received value: {chunk_size}")
        self.chunk_size = chunk_size
        
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative. Received value: {chunk_overlap}")
        self.chunk_overlap = chunk_overlap  

    def split_text(self, text: str) -> List[str]:
        
        """
        Splits the provided text into chunks.
        
        Splits based on the specified split_by, chunk size, and chunk overlap settings.

        Args:
            text (str): The text to split.

        Returns:
            List[str]: A list of text chunks.
        """
        
        splits = self._split_text(text, self.split_by)
        chunks = self._concatenate_splits(splits, self.chunk_size, self.chunk_overlap)
        return chunks

    def call(self, documents: List[Document]) -> List[Document]:
        """
        Process the splitting task on a list of documents.
        
        Processes a list of documents, splitting each document's text according to the configured
        split_by, chunk size, and chunk overlap.

        Args:
            documents (List[Document]): A list of Document objects to process.
            
        Returns:
            List[Document]: A list of new Document objects, each containing a chunk of text from the original documents.

        Raises:
            TypeError: If 'documents' is not a list or contains non-Document objects.
            ValueError: If any document's text is None.
        """
        
        if not isinstance(documents, list) or any(not isinstance(doc, Document) for doc in documents):
            raise TypeError("Input should be a list of Documents.")

        document_iterator = tqdm(documents, desc="Splitting Documents")
        
        split_docs = []
        for doc in document_iterator:
            
            if doc.text is None:
                raise ValueError(f"Text should not be None. Doc id: {doc.id}")
                
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
        
    def _split_text(
        self, text: str, split_by: str) -> List[str]:
        """Perform the actual splitting of text using the specified split_by."""
        # get the separator
        separator = separators[split_by]
        
        # for each piece of text, break into smaller splits 
        splits = text.split(separator)
        
        # separators will be added in the end of the split, except the last split
        splits = self._add_separator(splits, separator)
        return splits

    def _add_separator(self, splits: List[str], separator: str):
        """operate how split_by separator should be added back to each split here
        
        Adds the split_by separator to the end of each split except the last one, reforming the original text structure."""
        for i in range(len(splits) - 1):
            splits[i] += separator
        return splits
        
    def _concatenate_splits(
        self, splits: List[str], chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        """
        Concatenates split text chunks based on the specified chunk size and overlap.
        """
        # chunks = []
        # windowed_splits = windowed(splits, n=chunk_size, step=chunk_size - chunk_overlap)
        # for split in windowed_splits:
        #     current_splits = [unit for unit in split if unit is not None]
        #     windowed_text = "".join(current_splits)
        #     if len(windowed_text) > 0:
        #         chunks.append(windowed_text)
        # return chunks
        chunks = []
        start_index = 0
        end_index = start_index + chunk_size
        while end_index < len(splits):
            # filter out None to avoid join error
            current_splits = [unit for unit in splits[start_index:end_index] if unit is not None]
            chunk = "".join(current_splits)
            chunks.append(chunk)
            # update the next start, end pointers, step size = chunk_size - chunk_overlap
            start_index += chunk_size - chunk_overlap
            end_index = start_index + chunk_size
            # when the end point exceed the splits length, exit
            
        # if there's any content between start pointer and the end of the split, it should be included in the last chunk
        # process the last chunk if the len(last_chunk) > 0, if last_chunk="", ignore it.
        if start_index < len(splits):
            last_chunk = "".join(splits[start_index:])
            if len(last_chunk) > 0:
                chunks.append(last_chunk)
            
        return chunks

    def extra_repr(self) -> str:
        s = f"split_by={self.split_by}, chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}"
        return s