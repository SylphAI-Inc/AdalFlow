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


@dataclass
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


EmbedderInputType = Union[str, Sequence[str]]
EmbedderOutputType = EmbedderOutput

BatchEmbedderInputType = EmbedderInputType
BatchEmbedderOutputType = List[EmbedderOutputType]


@dataclass
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


@dataclass
class Document(DataClass):
    __doc__ = r"""A text container with optional metadata and vector representation.

    It is the data structure to support functions like Retriever, DocumentSplitter, and used with LocalDB.
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
        """Create a Document object from a dictionary.

        Example:

        .. code-block :: python

            doc = Document.from_dict({
                "id": "123",
                "text": "Hello world",
                "meta_data": {"title": "Greeting"}
            })
        """

        doc = doc.copy()
        assert "meta_data" in doc, "meta_data is required"
        assert "text" in doc, "text is required"
        if "estimated_num_tokens" not in doc:
            tokenizer = Tokenizer()
            doc["estimated_num_tokens"] = tokenizer.count_tokens(doc["text"])
        if "id" not in doc or not doc["id"]:
            doc["id"] = uuid.uuid4()

        return super().from_dict(doc)

    def __repr__(self):
        """Custom repr method to truncate the text to 100 characters and vector to 10 floats."""
        max_chars_to_show = 100
        truncated_text = (
            self.text[:max_chars_to_show] + "..."
            if len(self.text) > max_chars_to_show
            else self.text
        )
        truncated_vector = (
            f"len: {len(self.vector)}" if len(self.vector) else self.vector
        )
        return (
            f"Document(id={self.id}, text={truncated_text!r}, meta_data={self.meta_data}, "
            f"vector={truncated_vector!r}, parent_doc_id={self.parent_doc_id}, order={self.order}, "
            f"score={self.score})"
        )


RetrieverQueryType = TypeVar("RetrieverQueryType", contravariant=True)
RetrieverStrQueryType = str
RetrieverQueriesType = Union[RetrieverQueryType, Sequence[RetrieverQueryType]]
RetrieverStrQueriesType = Union[str, Sequence[RetrieverStrQueryType]]

RetrieverDocumentType = TypeVar("RetrieverDocumentType", contravariant=True)
RetrieverStrDocumentType = str  # for text retrieval
RetrieverDocumentsType = Sequence[RetrieverDocumentType]


@dataclass
class RetrieverOutput(DataClass):
    __doc__ = r"""Save the output of a single query in retrievers.

    It is up to the subclass of Retriever to specify the type of query and document.
    """

    doc_indices: List[int] = field(metadata={"desc": "List of document indices"})
    doc_scores: Optional[List[float]] = field(
        default=None, metadata={"desc": "List of document scores"}
    )
    query: Optional[RetrieverQueryType] = field(
        default=None, metadata={"desc": "The query used to retrieve the documents"}
    )
    documents: Optional[List[RetrieverDocumentType]] = field(
        default=None, metadata={"desc": "List of retrieved documents"}
    )


RetrieverOutputType = List[RetrieverOutput]  # so to support multiple queries at once


@dataclass
class UserQuery:
    query_str: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AssistantResponse:
    response_str: str
    metadata: Optional[Dict[str, Any]] = None


# There could  more other roles in a multi-party conversation. We might consider in the future.
@dataclass
class DialogTurn(DataClass):
    __doc__ = r"""A turn consists of a user query and the assistant response.

    The dataformat is designed to fit into a relational database, where each turn is a row.
    Use `session_id` to group the turns into a dialog session with the `order` field and
    `user_query_timestamp` and `assistant_response_timestamp` to order the turns.

    Args:

        id (str): The unique id of the turn.
        user_id (str, optional): The unique id of the user.
        session_id (str, optional): The unique id of the dialog session.
        order (int, optional): The order of the turn in the dialog session, starts from 0.
        user_query (UserQuery, optional): The user query in the turn.
        assistant_response (AssistantResponse, optional): The assistant response in the turn.
        user_query_timestamp (datetime, optional): The timestamp of the user query.
        assistant_response_timestamp (datetime, optional): The timestamp of the assistant response.
        metadata (Dict[str, Any], optional): Additional metadata.

    Examples:

        - User: Hi, how are you?
        - Assistant: I'm fine, thank you!
        DialogTurn(id=uuid4(), user_query=UserQuery("Hi, how are you?"), assistant_response=AssistantResponse("I'm fine, thank you!"))
    """

    id: str = field(
        default_factory=lambda: str(uuid.uuid4()),
        metadata={"desc": "The unique id of the turn"},
    )
    user_id: Optional[str] = field(
        default=None, metadata={"desc": "The unique id of the user"}
    )
    session_id: Optional[str] = field(
        default=None, metadata={"desc": "The unique id of the dialog session"}
    )
    order: Optional[int] = field(
        default=None,
        metadata={"desc": "The order of the turn in the Dialog Session, starts from 0"},
    )

    user_query: Optional[UserQuery] = field(
        default=None, metadata={"desc": "The user query in the turn"}
    )
    assistant_response: Optional[AssistantResponse] = field(
        default=None, metadata={"desc": "The assistant response in the turn"}
    )
    user_query_timestamp: Optional[datetime] = field(
        default_factory=datetime.now,
        metadata={"desc": "The timestamp of the user query"},
    )
    assistant_response_timestamp: Optional[datetime] = field(
        default_factory=datetime.now,
        metadata={"desc": "The timestamp of the assistant response"},
    )
    metadata: Optional[Dict[str, Any]] = field(
        default=None, metadata={"desc": "Additional metadata"}
    )
    vector: Optional[List[float]] = field(
        default=None,
        metadata={"desc": "The vector representation of the dialog turn"},
    )

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
    __doc__ = r"""A dialog session manages the dialog turns in a whole conversation as a session.

    This class is mainly used in-memory for the dialog system/app to manage active conversations.
    You won't need this class for past conversations which have already been persisted in a database as a form of
    record or history.
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
