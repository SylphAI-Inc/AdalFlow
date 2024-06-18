"""
Functional data classes to support functional components like Generator, Retriever, and Assistant.
"""

from enum import Enum, auto
from typing import List, Dict, Any, Optional, Union, Generic, TypeVar, Sequence
from collections import OrderedDict
from dataclasses import (
    dataclass,
    field,
    InitVar,
)
from uuid import UUID
from datetime import datetime
import uuid
import logging

from lightrag.core.base_data_class import DataClass
from lightrag.core.tokenizer import Tokenizer
from lightrag.core.functional import is_normalized
from lightrag.components.model_client import (
    CohereAPIClient,
    TransformersClient,
    AnthropicAPIClient,
    GroqAPIClient,
    OpenAIClient,
)


logger = logging.getLogger(__name__)

T_co = TypeVar("T_co", covariant=True)


class ModelType(Enum):
    EMBEDDER = auto()
    LLM = auto()
    RERANKER = auto()  # ranking model
    UNDEFINED = auto()


@dataclass
class ModelClientType:
    COHERE = CohereAPIClient
    TRANSFORMERS = TransformersClient
    ANTHROPIC = AnthropicAPIClient
    GROQ = GroqAPIClient
    OPENAI = OpenAIClient


# TODO: define standard required outputs
def get_model_args(model_type: ModelType) -> List[str]:
    r"""Get the required keys in model_kwargs for a specific model type.

    note:
    If your model inference sdk uses different keys, you need to convert them to the standard keys here in their specifc ModelClient.

    Args:
        model_type (ModelType): The model type

    Returns:
        List[str]: The required keys in model_kwargs
    """
    if model_type == ModelType.EMBEDDER:
        return ["model"]
    elif model_type == ModelType.LLM:
        return ["model"]
    elif model_type == ModelType.RERANKER:
        return ["model", "top_k", "documents", "query"]
    else:
        return []


@dataclass
class Embedding:
    """
    Container for a single embedding.

    In sync with api spec, same as openai/types/embedding.py
    """

    embedding: List[float]
    index: Optional[int]  # match with the index of the input, in case some are missing


@dataclass
class Usage:
    """
    In sync with api spec, same as openai/types/create_embedding_response.py
    """

    prompt_tokens: int
    total_tokens: int


class EmbedderOutput(DataClass):
    __doc__ = r"""Container to hold the response from an Embedder model. Only Per-batch.

    Data standard for Embedder model output to interact with other components.
    Batch processing is often available, thus we need a list of Embedding objects.
    """

    data: List[Embedding] = field(
        default_factory=list, metadata={"desc": "List of embeddings"}
    )
    model: Optional[str] = field(default=None, metadata={"desc": "Model name"})
    usage: Optional[Usage] = field(default=None, metadata={"desc": "Usage tracking"})
    error: Optional[str] = field(default=None, metadata={"desc": "Error message"})
    raw_response: Optional[Any] = field(
        default=None, metadata={"desc": "Raw response"}
    )  # only used if error
    input: Optional[List[str]] = field(default=None, metadata={"desc": "Input text"})

    @property
    def length(self) -> int:
        return len(self.data) if self.data and isinstance(self.data, Sequence) else 0

    @property
    def embedding_dim(self) -> int:
        r"""The dimension of the embedding, assuming all embeddings have the same dimension.

        Returns:
            int: The dimension of the embedding, -1 if no embedding is available
        """
        return (
            len(self.data[0].embedding) if self.data and self.data[0].embedding else -1
        )

    @property
    def is_normalized(self) -> bool:
        r"""Check if the embeddings are normalized to unit vectors.

        Returns:
            bool: True if the embeddings are normalized, False otherwise
        """
        return (
            is_normalized(self.data[0].embedding)
            if self.data and self.data[0].embedding
            else False
        )


EmbedderOutputType = EmbedderOutput
BatchEmbedderOutputType = List[EmbedderOutput]


class GeneratorOutput(DataClass, Generic[T_co]):
    __doc__ = r"""
    The output data class for the Generator component.
    We can not control its output 100%, so we use this to track the error_message and
    allow the raw string output to be passed through.

    (1) When model predict and output processors are both without error,
    we have data as the final output, error as None.
    (2) When either model predict or output processors have error,
    we have data as None, error as the error message.

    Raw_response will depends on the model predict.
    """

    data: T_co = field(
        default=None,
        metadata={"desc": "The final output data potentially after output parsers"},
    )
    error: Optional[str] = field(
        default=None,
        metadata={"desc": "Error message if any"},
    )
    usage: Optional[Usage] = field(default=None, metadata={"desc": "Usage tracking"})
    raw_response: Optional[str] = field(
        default=None, metadata={"desc": "Raw string response from the model"}
    )


GeneratorOutputType = GeneratorOutput[Any]


class Document(DataClass):
    r"""A text container with optional metadata and vector representation.
    It is the data structure to support functions like Retriever, DocumentSplitter, and LocalDocumentDB.
    """

    text: str = field(metadata={"desc": "The main text"})

    meta_data: Optional[Dict[str, Any]] = field(
        default=None, metadata={"desc": "Metadata for the document"}
    )
    # can save data for filtering at retrieval time too
    vector: List[float] = field(
        default_factory=list,
        metadata={"desc": "The vector representation of the document"},
    )
    # the vector representation of the document

    id: Optional[str] = field(
        default_factory=lambda: str(uuid.uuid4()), metadata={"desc": "Unique id"}
    )  # unique id of the document
    order: Optional[int] = field(
        default=None,
        metadata={"desc": "Order of the chunked document in the original document"},
    )

    score: Optional[float] = field(
        default=None,
        metadata={"desc": "Score of the document, likely used in retrieval output"},
    )
    parent_doc_id: Optional[Union[str, UUID]] = field(
        default=None, metadata={"desc": "id of the Document where the chunk is from"}
    )

    estimated_num_tokens: Optional[int] = field(
        default=None,
        metadata={
            "desc": "Estimated number of tokens in the text, useful for cost estimation"
        },
    )

    def __post_init__(self):
        if self.estimated_num_tokens is None and self.text:
            tokenizer = Tokenizer()
            self.estimated_num_tokens = tokenizer.count_tokens(self.text)

    @classmethod
    def from_dict(cls, doc: Dict):
        doc = doc.copy()
        assert "meta_data" in doc, "meta_data is required"
        assert "text" in doc, "text is required"
        if "estimated_num_tokens" not in doc:
            tokenizer = Tokenizer()
            doc["estimated_num_tokens"] = tokenizer.count_tokens(doc["text"])
        if "id" not in doc or not doc["id"]:
            doc["id"] = uuid.uuid4()

        return super().from_dict(doc)

    def __repr__(self) -> str:
        max_text_len_to_show: int = 400
        repr_str = "Document("
        if self.id:
            repr_str += f"id={self.id}, "
        if self.text:
            repr_str += f"text={self.text[0:max_text_len_to_show]} "
            if len(self.text) > max_text_len_to_show:
                repr_str += "..."
            repr_str += ", "
        if self.meta_data:
            repr_str += f"meta_data={self.meta_data}, "
        if self.estimated_num_tokens:
            repr_str += f"estimated_num_tokens={self.estimated_num_tokens}, "

        if self.vector:
            repr_str += f"vector={self.vector[0:10]}..., "
        if self.score:
            repr_str += f"score={self.score}, "
        if self.parent_doc_id:
            repr_str += f"parent_doc_id={self.parent_doc_id}, "
        repr_str = repr_str[:-2] + ")"
        return repr_str

    def __str__(self):
        return self.__repr__()


class RetrieverOutput(DataClass):
    __doc__ = r"""Save the output of a single query in retrievers."""

    doc_indices: List[int] = field(metadata={"desc": "List of document indices"})
    doc_scores: Optional[List[float]] = field(
        default=None, metadata={"desc": "List of document scores"}
    )
    query: Optional[str] = field(
        default=None, metadata={"desc": "The query used to retrieve the documents"}
    )
    documents: Optional[List[Document]] = field(
        default=None, metadata={"desc": "List of retrieved documents"}
    )


RetrieverInputStrType = Union[str, Sequence[str]]
RetrieverInputType = TypeVar("RetrieverInputType", contravariant=True)
RetrieverDocumentType = TypeVar("RetrieverDocumentType", contravariant=True)
RetrieverDocumentsType = Sequence[Any]
# it is up the the subclass to decide the type of the documents
RetrieverOutputType = List[RetrieverOutput]  # so to support multiple queries at once

# different retriever may support different input data type


@dataclass
class UserQuery:
    query_str: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AssistantResponse:
    response_str: str
    metadata: Optional[Dict[str, Any]] = None


# TODO: the data class can also be a component?
@dataclass
class DialogTurn:
    r"""A turn consists of a user query and the assistant response, \
    or potentially with more other roles in a multi-party conversation.

    Here we only consider the user query and the assistant response. If you want multiple parties
    you can extend this class or create a new class.
    Examples:
    - User: Hi, how are you?
    - Assistant: I'm fine, thank you!
    DialogTurn(id=uuid4(), user_query=UserQuery("Hi, how are you?"), assistant_response=AssistantResponse("I'm fine, thank you!"))
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    order: Optional[int] = (
        None  # the order of the turn in the Dialog Session, starts from 0
    )
    user_query: Optional[UserQuery] = None
    assistant_response: Optional[AssistantResponse] = None
    user_query_timestamp: Optional[datetime] = None
    assistant_response_timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None  # additional metadata

    def set_user_query(
        self, user_query: UserQuery, user_query_timestamp: Optional[datetime] = None
    ):
        self.user_query = user_query
        if not user_query_timestamp:
            user_query_timestamp = datetime.now()
        self.user_query_timestamp = user_query_timestamp

    def set_assistant_response(
        self,
        assistant_response: AssistantResponse,
        assistant_response_timestamp: Optional[datetime] = None,
    ):
        self.assistant_response = assistant_response
        if not assistant_response_timestamp:
            assistant_response_timestamp = datetime.now()
        self.assistant_response_timestamp = assistant_response_timestamp


@dataclass
class DialogSession:
    r"""A dialog session consists of multiple dialog turns, \
    and potentially with more other roles in a multi-party conversation.

    Here we only consider the dialog turns. If you want multiple parties
    you can extend this class or create a new class.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # the id of the session
    user_id: Optional[str] = None
    dialog_turns: OrderedDict[int, DialogTurn] = field(default_factory=OrderedDict)
    # int is the order of the turn, starts from 0
    metadata: Optional[Dict[str, Any]] = None
    session_start_timestamp: Optional[datetime] = field(default_factory=datetime.now)

    # InitVar type annotation is used for parameters that are used in __post_init__
    # but not meant to be fields in the dataclass.
    dialog_turns_input: InitVar[
        Optional[Union[OrderedDict[int, DialogTurn], List[DialogTurn]]]
    ] = None

    def __post_init__(
        self,
        dialog_turns_input: Optional[
            Union[OrderedDict[int, DialogTurn], List[DialogTurn]]
        ] = None,
    ):
        if dialog_turns_input:
            if isinstance(dialog_turns_input, list):
                # Assume the list is of DialogTurn objects and needs to be added to an OrderedDict
                for order, dialog_turn in enumerate(dialog_turns_input):
                    self.append_dialog_turn(dialog_turn)
            elif isinstance(dialog_turns_input, OrderedDict):
                self.dialog_turns = dialog_turns_input
            else:
                raise ValueError(
                    "dialog_turns should be a list of DialogTurn or an OrderedDict"
                )

    def get_next_order(self):
        return len(self.dialog_turns)

    def append_dialog_turn(self, dialog_turn: DialogTurn):
        next_order = self.get_next_order()
        if dialog_turn.order is None:
            dialog_turn.order = next_order
        else:
            assert dialog_turn.order == next_order, f"order should be {next_order}"
        self.dialog_turns[next_order] = dialog_turn

    def get_dialog_turns(self) -> OrderedDict[int, DialogTurn]:
        return self.dialog_turns

    def get_chat_history_str(self) -> str:
        chat_history_str = ""
        for order, dialog_turn in self.dialog_turns.items():
            chat_history_str += f"User: {dialog_turn.user_query.query_str}\n"
            chat_history_str += (
                f"Assistant: {dialog_turn.assistant_response.response_str}\n"
            )
        return chat_history_str

    def delete_dialog_turn(self, order: int):
        self.dialog_turns.pop(order)

    def update_dialog_turn(self, order: int, dialog_turn: DialogTurn):
        self.dialog_turns[order] = dialog_turn
