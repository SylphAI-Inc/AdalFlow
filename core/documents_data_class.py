r"""Data classes to be consumed by retriever component.
(1) It acts as interface to the database, local or cloud.
(2) It works in tandem with the vectorizer component.
"""

from typing import List, Dict, Any, Optional, Union
from uuid import UUID
import uuid
from core.tokenizer import Tokenizer


##############################################
# Key data structures for RAG
# TODO: visualize the data structures
##############################################
class Document:
    meta_data: dict  # can save data for filtering at retrieval time too
    text: str
    id: Optional[Union[str, UUID]] = (
        None  # if the file name is unique, its better to use it as id instead of UUID
    )
    estimated_num_tokens: Optional[int] = (
        None  # useful for cost and chunking estimation
    )

    def __init__(
        self,
        meta_data: dict,
        text: str,
        id: Optional[Union[str, UUID]] = None,
        estimated_num_tokens: Optional[int] = None,
    ):
        self.meta_data = meta_data
        self.text = text
        self.id = id
        self.estimated_num_tokens = estimated_num_tokens

    @staticmethod
    def from_dict(doc: Dict):
        assert "meta_data" in doc, "meta_data is required"
        assert "text" in doc, "text is required"
        if "estimated_num_tokens" not in doc:
            tokenizer = Tokenizer()
            doc["estimated_num_tokens"] = tokenizer.count_tokens(doc["text"])
        if "id" not in doc:
            doc["id"] = uuid.uuid4()

        return Document(**doc)

    def __repr__(self) -> str:
        return f"Document(id={self.id}, meta_data={self.meta_data}, text={self.text[0:50]}, estimated_num_tokens={self.estimated_num_tokens})"

    def __str__(self):
        return self.__repr__()


class Chunk:
    vector: List[float]
    text: str
    order: Optional[int] = (
        None  # order of the chunk in the document. Llama index uses RelatedNodeInfo which is an overkill
    )

    doc_id: Optional[Union[str, UUID]] = (
        None  # id of the Document where the chunk is from
    )
    id: Optional[Union[str, UUID]] = None
    estimated_num_tokens: Optional[int] = None
    score: Optional[float] = None  # used in retrieved output
    meta_data: Optional[Dict] = (
        None  # only when the above fields are not enough or be used for metadata filtering
    )

    def __init__(
        self,
        vector: List[float],
        text: str,
        order: Optional[int] = None,
        doc_id: Optional[Union[str, UUID]] = None,
        id: Optional[Union[str, UUID]] = None,
        estimated_num_tokens: Optional[int] = None,
        meta_data: Optional[Dict] = None,
    ):
        self.vector = vector if vector else []
        self.text = text
        self.order = order
        self.doc_id = doc_id
        self.id = id if id else uuid.uuid4()
        self.meta_data = meta_data

        self.estimated_num_tokens = estimated_num_tokens if estimated_num_tokens else 0
        # estimate the number of tokens
        if not self.estimated_num_tokens:
            tokenizer = Tokenizer()
            self.estimated_num_tokens = tokenizer.count_tokens(self.text)

    def __repr__(self) -> str:
        return f"Chunk(id={self.id}, doc_id={self.doc_id}, order={self.order}, text={self.text}, vector={self.vector[0:5]}, estimated_num_tokens={self.estimated_num_tokens}, score={self.score})"

    def __str__(self):
        return self.__repr__()


class RetrieverOutput:
    """
    Retrieved result per query
    """

    chunks: List[Chunk]
    query: Optional[str] = None

    def __init__(self, chunks: List[Chunk], query: Optional[str] = None):
        self.chunks = chunks
        self.query = query

    def __repr__(self) -> str:
        return f"RetrieverOutput(chunks={self.chunks[0:5]}, query={self.query})"

    def __str__(self):
        return self.__repr__()
