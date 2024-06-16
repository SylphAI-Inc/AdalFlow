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
from tqdm import tqdm

from lightrag.core.component import Component
from lightrag.core.types import Document

# TODO:
# More splitters such as PDF/JSON/HTML Splitter can be built on TextSplitter.

DocumentSplitterInputType = List[Document]
DocumentSplitterOutputType = List[Document]

# customizable seperators map
separators = {"page": "\f", "passage": "\n\n", "word": " ", "sentence": "."}

DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 20

class TextSplitter(Component):
    """        
    Text Splitter for Processing and Chunking Documents

    * **Overview**
    The ``TextSplitter`` supports plain text splitting. It first utilizes a ``split_by`` argument to specify the 
    text-splitting criterion. The long text will get broken down into a list of shorter texts. 
    Then we create a sliding window with length=``chunk_size``. It moves at step=``chunk_size``-``chunk_overlap``.
    The texts inside each window will get concatenated to a small chunk. 
    
    * **Definitions**
    ``separators``: A dictionary that maps ``split_by`` keys to their corresponding separator strings. 
    
    ``split_by``: A parameter that selects the key from the ``separators`` dictionary to determine how the text is split. It defines the rule or boundary for splitting text.
    
    * **Splitting Details**
    The ``TextSplitter`` utilizes Python's ``str.split(separator)`` method. Valid options include "word", "sentence", "page", and "passage". 
    Developers can refer to ``separators`` = {"page": "\f", "passage": "\n\n", "word": " ", "sentence": "."} for exact points of text division.
    Separators mapping relationship: separators[split_by]=separator.

    * **Customization**
    You can also customize the separators for different needs. For example, by defining separators = {"question": "?"} 
    and setting ``split_by``="question", the document will be split at each question mark, ideal for processing text structured 
    as a series of questions.
    
    .. note::
        Typically the split texts will be embedded and potentially get retrieved. 
        Developers need to determine how to assign text to each data trunk for the embedding and retrieval tasks.
        The ``TextSplitter`` ``split_by`` examples:
        - "word": Splits the text at every space (' '), treating spaces as the boundaries between words.
        - "sentence": Splits the text at every period followed by a space ('. '), treating these as the ends of sentences.
        - "page": Splits the text at form feed characters ('\\f'), which are often used to represent page breaks in documents.
        - "passage": Splits the text at double newline characters ('\\n\\n'), useful for distinguishing between paragraphs or sections.
    
    * **Concatenating Details**
    The TextSplitter then reattaches the specified separator to each piece of the split text, except for the last segment.
    This approach maintains the original spacing and punctuation, which is critical in contexts like natural language processing where text formatting can impact interpretations and outcomes.
    E.g. "hello world!" split by "word" will be kept as "hello " and "world!"
    
    * **Usage Examples**
    - This functionality is ideal for segmenting texts into sentences, words, pages, or passages, which can then be processed further for NLP applications.
    
    - To handle PDF content, developers need to first extract the text using tools like `PyPDF2` or `PDFMiner` before splitting.
    
    Example:
        .. code-block:: python
        
            from lightrag.core.text_splitter import TextSplitter
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
        batch_size: int = 1000
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
            batch_size (int): The size of documents to process in each batch.
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
        
        self.batch_size = batch_size

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
        Process the splitting task on a list of documents in batch.
        
        Batch processes a list of documents, splitting each document's text according to the configured
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
        
        split_docs = []
        # Using range and batch_size to create batches
        for start_idx in tqdm(range(0, len(documents), self.batch_size), desc="Splitting Documents in Batches"):
            batch_docs = documents[start_idx:start_idx + self.batch_size]
            
            for doc in batch_docs:
                if not isinstance(doc, Document):
                    raise TypeError(f"Each item in documents should be an instance of Document, but got {type(doc).__name__}.")

                if doc.text is None:
                    raise ValueError(f"Text should not be None. Doc id: {doc.id}")

                text_splits = self.split_text(doc.text)
                meta_data = deepcopy(doc.meta_data)

                split_docs.extend([
                    Document(
                        text=txt,
                        meta_data=meta_data,
                        parent_doc_id=f"{doc.id}",
                        order=i,
                        vector=[],
                    )
                    for i, txt in enumerate(text_splits)
                ])
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
        chunks = []
        start_index = 0
        end_index = start_index + chunk_size
        while end_index < len(splits):
            # filter out None to avoid join error
            current_splits = [unit for unit in splits[start_index:end_index]]
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