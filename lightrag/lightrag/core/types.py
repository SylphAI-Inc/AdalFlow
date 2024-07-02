"""Functional data classes to support functional components like Generator, Retriever, and Assistant."""

from enum import Enum, auto
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Union,
    Generic,
    TypeVar,
    Sequence,
    Literal,
    Callable,
    Awaitable,
)
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

from lightrag.core.base_data_class import DataClass, required_field
from lightrag.core.tokenizer import Tokenizer
from lightrag.core.functional import (
    is_normalized,
    generate_function_call_expression_from_callable,
)
from lightrag.components.model_client import (
    CohereAPIClient,
    TransformersClient,
    AnthropicAPIClient,
    GroqAPIClient,
    OpenAIClient,
)


logger = logging.getLogger(__name__)

T_co = TypeVar("T_co", covariant=True)


#######################################################################################
# Data modeling for ModelClient
######################################################################################
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


#######################################################################################
# Data modeling for Embedder component
######################################################################################
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


#######################################################################################
# Data modeling for Generator component
######################################################################################
@dataclass
class TokenLogProb:
    r"""similar to openai.ChatCompletionTokenLogprob"""

    token: str
    logprob: float


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
    )  # parsed from model client response
    metadata: Optional[Dict[str, object]] = field(
        default=None, metadata={"desc": "Additional metadata"}
    )


GeneratorOutputType = GeneratorOutput[object]

#######################################################################################
# Data modeling for Retriever component
######################################################################################

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


#######################################################################################
# Data modeling for function calls
######################################################################################
AsyncCallable = Callable[..., Awaitable[Any]]


@dataclass
class FunctionDefinition(DataClass):
    __doc__ = r"""The data modeling of a function definition, including the name, description, and parameters."""

    func_name: str = field(metadata={"desc": "The name of the tool"})
    func_desc: Optional[str] = field(
        default=None, metadata={"desc": "The description of the tool"}
    )
    func_parameters: Dict[str, object] = field(
        default_factory=dict, metadata={"desc": "The schema of the parameters"}
    )

    def fn_schema_str(self, type: Literal["json", "yaml"] = "json") -> str:
        r"""Get the function definition str to be used in the prompt.

        You should also directly use :meth:`to_json` and :meth:`to_yaml` to get the schema in JSON or YAML format.
        """
        if type == "json":
            return self.to_json()
        elif type == "yaml":
            return self.to_yaml()
        else:
            raise ValueError(f"Unsupported type: {type}")


@dataclass
class Function(DataClass):
    __doc__ = r"""The data modeling of a function call, including the name and keyword arguments.

    You can use the exclude in :meth:`to_json` and :meth:`to_yaml` to exclude the `thought` field if you do not want to use chain-of-thought pattern.

    Example:

    .. code-block:: python

        # assume the function is added in a context_map
        # context_map = {"add": add}

        def add(a, b):
            return a + b

        # call function add with arguments 1 and 2
        fun = Function(name="add", kwargs={"a": 1, "b": 2})
        # evaluate the function
        result = context_map[fun.name](**fun.kwargs)

        # or call with positional arguments
        fun = Function(name="add", args=[1, 2])
        result = context_map[fun.name](*fun.args)
    """
    thought: Optional[str] = field(
        default=None, metadata={"desc": "Why the function is called"}
    )
    name: str = field(default="", metadata={"desc": "The name of the function"})
    args: Optional[List[object]] = field(
        default_factory=list,
        metadata={"desc": "The positional arguments of the function"},
    )
    kwargs: Optional[Dict[str, object]] = field(
        default_factory=dict,
        metadata={"desc": "The keyword arguments of the function"},
    )


@dataclass
class FunctionExpression(DataClass):
    __doc__ = r"""The data modeling of a function expression for a call, including the name and arguments.

    Example:

    .. code-block:: python

        def add(a, b):
            return a + b

        # call function add with positional arguments 1 and 2
        fun_expr = FunctionExpression(action="add(1, 2)")
        # evaluate the expression
        result = eval(fun_expr.action)
        print(result)
        # Output: 3

        # call function add with keyword arguments
        fun_expr = FunctionExpression(action="add(a=1, b=2)")
        result = eval(fun_expr.action)
        print(result)
        # Output: 3

    Why asking LLM to generate function expression (code snippet) for a function call?
    - It is more efficient/compact to call a function.
    - It is more flexible.
        (1) for the full range of Python expressions, including arithmetic operations, nested function calls, and more.
        (2) allow to pass variables as arguments.
    - Ease of parsing using ``ast`` module.

    The benefits are less failed function calls.
    """
    thought: Optional[str] = field(
        default=None, metadata={"desc": "Why the function is called"}
    )
    action: str = field(
        default_factory=required_field,
        # metadata={"desc": "FuncName(<args>, <kwargs>)"},
        metadata={
            "desc": """FuncName(<kwargs>) \
                Valid function call expression. \
                Example: "FuncName(a=1, b=2)" \
                Follow the data type specified in the function parameters.\
                e.g. for Type object with x,y properties, use "ObjectType(x=1, y=2)"""
        },
    )

    @classmethod
    def from_function(
        cls,
        func: Union[Callable[..., Any], AsyncCallable],
        thought: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "FunctionExpression":
        r"""Create a FunctionExpression object from a function.

        Args:
            fun (Union[Callable[..., Any], AsyncCallable]): The function to be converted

        Returns:
            FunctionExpression: The FunctionExpression object

        Usage:
        1. Create a FunctionExpression object from a function call:
        2. use :meth:`to_json` and :meth:`to_yaml` to get the schema in JSON or YAML format.
        3. This will be used as an example in prompt showing LLM how to call the function.
        """
        try:
            action = generate_function_call_expression_from_callable(
                func, *args, **kwargs
            )
        except Exception as e:
            logger.error(f"Error generating function expression: {e}")
            raise ValueError(f"Error generating function expression: {e}")
        return cls(action=action, thought=thought)


# saves the output of a function tool.


@dataclass
class FunctionOutput(DataClass):
    __doc__ = (
        r"""The output of a tool, which could be a function, a class, or a module."""
    )
    name: Optional[str] = field(
        default=None, metadata={"desc": "The name of the function"}
    )
    input: Optional[Union[Function, FunctionExpression]] = field(
        default=None, metadata={"desc": "The Function or FunctionExpression object"}
    )
    parsed_input: Optional[Function] = field(
        default=None,
        metadata={
            "desc": "The parsed Function object if the input is FunctionExpression"
        },
    )
    output: Optional[object] = field(
        default=None, metadata={"desc": "The output of the function execution"}
    )
    error: Optional[str] = field(
        default=None, metadata={"desc": "The error message if any"}
    )


#######################################################################################
# Data modeling for agent component
######################################################################################
@dataclass
class StepOutput(DataClass):
    __doc__ = r"""The output of a single step in the agent."""
    step: int = field(
        default=0, metadata={"desc": "The order of the step in the agent"}
    )
    thought: Optional[str] = field(
        default="", metadata={"desc": "The thought of the agent in the step"}
    )
    action: str = field(
        default="", metadata={"desc": "The action of the agent in the step"}
    )
    fun_name: Optional[str] = field(
        default=None, metadata={"desc": "The function named parsed from action"}
    )
    fun_args: Optional[List[Any]] = field(
        default=None,
        metadata={"desc": "The function positional arguments parsed from action"},
    )
    fun_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"desc": "The function keyword arguments parsed from action"},
    )
    observation: Optional[str] = field(
        default=None, metadata={"desc": "The result of the action"}
    )

    def __str__(self):
        return f"Thought {self.step}: {self.thought}\nAction {self.step}: {self.action}\nObservation {self.step}: {self.observation}"


#######################################################################################
# Data modeling for data processing pipleline such as Text splitting and Embedding
######################################################################################
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


#######################################################################################
# Data modeling for dialog system
######################################################################################
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
    conversation_id: Optional[str] = field(
        default=None,
        metadata={"desc": "The unique id of the conversation it belongs to"},
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


# TODO: This part and the Memory class is still WIP, and will need more work in the future.
@dataclass
class Conversation:
    __doc__ = r"""A conversation manages the dialog turns in a whole conversation as a session.

    This class is mainly used in-memory for the dialog system/app to manage active conversations.
    You won't need this class for past conversations which have already been persisted in a database as a form of
    record or history.
    """

    id: str = field(
        default_factory=lambda: str(uuid.uuid4()),
        metadata={"desc": "The id of the conversation"},
    )  # the id of the conversation
    name: Optional[str] = field(
        default=None, metadata={"desc": "The name of the conversation"}
    )
    user_id: Optional[str] = field(
        default=None, metadata={"desc": "The id of the user"}
    )
    dialog_turns: OrderedDict[int, DialogTurn] = field(
        default_factory=OrderedDict, metadata={"desc": "The dialog turns"}
    )
    # int is the order of the turn, starts from 0
    metadata: Optional[Dict[str, Any]] = field(
        default=None, metadata={"desc": "Additional metadata"}
    )

    created_at: Optional[datetime] = field(
        default_factory=datetime.now,
        metadata={"desc": "The timestamp of the conversation creation"},
    )

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
