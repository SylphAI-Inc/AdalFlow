"""
Splitting texts is commonly used as a preprocessing step before embedding and retrieving texts.

We encourage you to process your data here and define your own embedding and retrieval methods. These methods can highly depend on the product environment and may extend beyond the scope of this library.

However, the following approaches are commonly shared:

* **Document Storage:** Define how to store the documents, both raw and chunked. For example, LlamaIndex uses `Document Stores <https://docs.llamaindex.ai/en/stable/module_guides/storing/docstores/>`_ to manage ingested document chunks.

* **Document Chunking:** Segment documents into manageable chunks suitable for further processing.

* **Vectorization:** Embed each chunk and store the resulting vectors in Vector Stores. For example, LLama index utilizes `Vector Stores <https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/>`_.

* **Retrieval:** Leverage vectors for context retrieval.
"""

from copy import deepcopy
from typing import List, Literal
from tqdm import tqdm
import logging

from lightrag.core.component import Component
from lightrag.core.types import Document
from lightrag.core.tokenizer import Tokenizer

# TODO:
# More splitters such as PDF/JSON/HTML Splitter can be built on TextSplitter.

log = logging.getLogger(__name__)

DocumentSplitterInputType = List[Document]
DocumentSplitterOutputType = List[Document]

# customizable seperators map
SEPARATORS = {
    "page": "\f",
    "passage": "\n\n",
    "word": " ",
    "sentence": ".",
    "token": "",
}

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 200

tokenizer = Tokenizer()


class TextSplitter(Component):
    """
    Text Splitter for Chunking Documents

    ``TextSplitter`` first utilizes ``split_by`` to specify the text-splitting criterion and breaks the long text into smaller texts.
    Then we create a sliding window with length= ``chunk_size``. It moves at step= ``chunk_size`` - ``chunk_overlap``.
    The texts inside each window will get merged to a smaller chunk. The generated chunks from the splitted text will be returned.

    **Splitting Types**

    ``TextSplitter`` supports 2 types of splitting.

    * **Type 1:** Specify the exact text splitting point such as space<" "> and periods<".">. It is intuitive, for example, split_by "word":

    ::

        "Hello, world!" -> ["Hello, " ,"world!"]

    * **Type 2:** Use :class:`tokenizer <lightrag.core.tokenizer.Tokenizer>`. It works as:

    ::

        "Hello, world!" -> ['Hello', ',', ' world', '!']

    This aligns with how models see text in the form of tokens (`Reference <https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb>`_),
    Tokenizer reflects the real token numbers the models take in and helps the developers control budgets.

    **Definitions**

    * **split_by** specifies the split rule, i.e. the smallest unit during splitting. We support ``"word"``, ``"sentence"``, ``"page"``, ``"passage"``, and ``"token"``. The splitter utilizes the corresponding separator from the ``SEPARATORS`` dictionary.
    For Type 1 splitting, we apply ``Python str.split()`` to break the text.

    * **SEPARATORS**: Maps ``split_by`` criterions to their exact text separators, e.g., spaces <" "> for "word" or periods <"."> for "sentence".

    .. note::
        For option ``token``, its separator is "" because we directly split by a tokenizer, instead of text point.

    * **chunk_size** is the the maximum number of units in each chunk.

    * **chunk_overlap** is the number of units that each chunk should overlap. Including context at the borders prevents sudden meaning shift in text between sentences/context, especially in sentiment analysis.


    * **Splitting Details**
    Type 1:
    The ``TextSplitter`` utilizes Python's ``str.split(separator)`` method.
    Developers can refer to

    .. code-block:: none

        {
            "page": "\\f",
            "passage": "\\n",
            "word": " ",
            "sentence": "."
        }
    for exact points of text division.

    .. note::
        Developers need to determine how to assign text to each data chunk for the embedding and retrieval tasks.

    Type 2:
    We implement a tokenizer using ``cl100k_base`` encoding that aligns with how models see text in the form of tokens.
    E.g. "tiktoken is great!" -> ["t", "ik", "token", " is", " great", "!"] This helps developers control the token usage and budget better.

    * **Merge Details**
    Type 1/Type 2 create a list of split texts. ``TextSplitter`` then reattaches the specified separator to each piece of the split text, except for the last segment.
    This approach maintains the original spacing and punctuation, which is critical in contexts like natural language processing where text formatting can impact interpretations and outcomes.
    E.g. "hello world!" split by "word" will be kept as "hello " and "world!"

    * **Customization**
    You can also customize the ``SEPARATORS``. For example, by defining ``SEPARATORS`` = {"question": "?"} and setting ``split_by`` = "question", the document will be split at each ``?``, ideal for processing text structured
    as a series of questions. If you need to customize :class:`tokenizer <lightrag.core.tokenizer.Tokenizer>`, please check `Reference <https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb>`_.

    * **Integration with Other Document Types**
    This functionality is ideal for segmenting texts into sentences, words, pages, or passages, which can then be processed further for NLP applications.
    For **PDFs**, developers will need to extract the text before using the splitter. Libraries like ``PyPDF2`` or ``PDFMiner`` can be utilized for this purpose.
    ``LightRAG``'s future implementations will introduce splitters for ``JSON``, ``HTML``, ``markdown``, and ``code``.

    Example:

    .. code-block:: python

        from lightrag.components.data_process.text_splitter import TextSplitter
        from lightrag.core.types import Document

        # Configure the splitter settings
        text_splitter = TextSplitter(
            split_by="word",
            chunk_size=5,
            chunk_overlap=1
        )

        # Example document
        doc = Document(
            text="Example text. More example text. Even more text to illustrate.",
            id="doc1"
        )

        # Execute the splitting
        splitted_docs = text_splitter.call(documents=[doc])

        for doc in splitted_docs:
            print(doc)

        # Output:
        # Document(id=44a8aa37-0d16-40f0-9ca4-2e25ae5336c8, text='Example text. More example text. ', meta_data=None, vector=[], parent_doc_id=doc1, order=0, score=None)
        # Document(id=ca0af45b-4f88-49b5-97db-163da9868ea4, text='text. Even more text to ', meta_data=None, vector=[], parent_doc_id=doc1, order=1, score=None)
        # Document(id=e7b617b2-3927-4248-afce-ec0fc247ac8b, text='to illustrate.', meta_data=None, vector=[], parent_doc_id=doc1, order=2, score=None)
    """

    def __init__(
        self,
        split_by: Literal["word", "sentence", "page", "passage", "token"] = "word",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        batch_size: int = 1000,
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

        self.split_by = split_by
        assert (
            split_by in SEPARATORS
        ), f"Invalid options for split_by. You must select from {list(SEPARATORS.keys())}."

        assert (
            chunk_overlap < chunk_size
        ), f"chunk_overlap can't be larger than or equal to chunk_size. Received chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}"

        assert (
            chunk_size > 0
        ), f"chunk_size must be greater than 0. Received value: {chunk_size}"
        self.chunk_size = chunk_size

        assert (
            chunk_overlap >= 0
        ), f"chunk_overlap must be non-negative. Received value: {chunk_overlap}"
        self.chunk_overlap = chunk_overlap

        self.batch_size = batch_size

        log.info(
            f"Initialized TextSplitter with split_by={self.split_by}, chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, batch_size={self.batch_size}"
        )

    def split_text(self, text: str) -> List[str]:
        """
        Splits the provided text into chunks.

        Splits based on the specified split_by, chunk size, and chunk overlap settings.

        Args:
            text (str): The text to split.

        Returns:
            List[str]: A list of text chunks.
        """
        log.info(
            f"Splitting text with split_by: {self.split_by}, chunk_size: {self.chunk_size}, chunk_overlap: {self.chunk_overlap}"
        )
        separator = SEPARATORS[self.split_by]
        splits = self._split_text_into_units(text, separator)
        log.info(f"Text split into {len(splits)} parts.")
        chunks = self._merge_units_to_chunks(
            splits, self.chunk_size, self.chunk_overlap, separator
        )
        log.info(f"Text merged into {len(chunks)} chunks.")
        return chunks

    def call(self, documents: DocumentSplitterInputType) -> DocumentSplitterOutputType:
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

        if not isinstance(documents, list) or any(
            not isinstance(doc, Document) for doc in documents
        ):
            log.error("Input should be a list of Documents.")
            raise TypeError("Input should be a list of Documents.")

        split_docs = []
        # Using range and batch_size to create batches
        for start_idx in tqdm(
            range(0, len(documents), self.batch_size),
            desc="Splitting Documents in Batches",
        ):
            batch_docs = documents[start_idx : start_idx + self.batch_size]

            for doc in batch_docs:
                if not isinstance(doc, Document):
                    log.error(
                        f"Each item in documents should be an instance of Document, but got {type(doc).__name__}."
                    )
                    raise TypeError(
                        f"Each item in documents should be an instance of Document, but got {type(doc).__name__}."
                    )

                if doc.text is None:
                    log.error(f"Text should not be None. Doc id: {doc.id}")
                    raise ValueError(f"Text should not be None. Doc id: {doc.id}")

                text_splits = self.split_text(doc.text)
                meta_data = deepcopy(doc.meta_data)

                split_docs.extend(
                    [
                        Document(
                            text=txt,
                            meta_data=meta_data,
                            parent_doc_id=f"{doc.id}",
                            order=i,
                            vector=[],
                        )
                        for i, txt in enumerate(text_splits)
                    ]
                )
        log.info(
            f"Processed {len(documents)} documents into {len(split_docs)} split documents."
        )
        return split_docs

    def _split_text_into_units(self, text: str, separator: str) -> List[str]:
        """Split text based on the specified separator."""
        if self.split_by == "token":
            splits = tokenizer.encode(text)
        else:
            splits = text.split(separator)
        log.info(f"Text split by '{separator}' into {len(splits)} parts.")
        return splits

    def _merge_units_to_chunks(
        self, splits: List[str], chunk_size: int, chunk_overlap: int, separator: str
    ) -> List[str]:
        """
        Merge split text chunks based on the specified chunk size and overlap.
        """
        chunks = []
        # we use a window to get the text for each trunk, the window size is chunk_size, step is chunk_size - chunk_overlap
        step = chunk_size - chunk_overlap
        idx = 0

        for idx in range(0, len(splits), step):
            # 1. if the window exceeds the list of splitted string, break and process the last chunk
            # 2. if the window ends exactly the same with the splits, then break and treat the splits[idx:len(splits)] as the last chunk
            if idx + chunk_size >= len(splits):
                break
            current_splits = splits[idx : idx + chunk_size]
            # add the separator between each unit and merge the string
            # this won't be the last chunk, so we need to add the separator at the end
            if self.split_by == "token":
                chunk = current_splits  # if token, then keep the original form
            else:
                chunk = separator.join(current_splits) + separator
            chunks.append(chunk)

        if idx < len(splits):
            if self.split_by == "token":
                last_chunk = splits[idx:]  # if token, then keep the original form
            else:
                last_chunk = separator.join(
                    splits[idx:]
                )  # if not token, then join into string
            if len(last_chunk) > 0:
                chunks.append(last_chunk)

        if self.split_by == "token":
            # decode each chunk here
            chunks = [tokenizer.decode(chunk) for chunk in chunks]

        log.info(f"Merged into {len(chunks)} chunks.")
        return chunks

    def _extra_repr(self) -> str:
        s = f"split_by={self.split_by}, chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}"
        return s


# test the execution llamaindex and langchain
