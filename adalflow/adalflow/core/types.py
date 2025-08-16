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
    Generator,
    AsyncGenerator,
    AsyncIterator,
    Coroutine,
    Iterable,
    Tuple,
)
from typing_extensions import TypeAlias
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
import json
from collections.abc import AsyncIterable

from adalflow.core.base_data_class import DataClass, required_field
from adalflow.core.tokenizer import Tokenizer
from adalflow.core.functional import (
    is_normalized,
    generate_function_call_expression_from_callable,
)

# Import OpenAI's ResponseStreamEvent for type alias

logger = logging.getLogger(__name__)

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")  # invariant type


#######################################################################################
# Data modeling for ModelClient
######################################################################################
class ModelType(Enum):
    __doc__ = r"""The type of the model, including Embedder, LLM, Reranker.
     It helps ModelClient identify the model type required to correctly call the model."""
    EMBEDDER = auto()
    LLM = auto()
    LLM_REASONING = auto()  # use reasoning model compatible to openai.responses
    RERANKER = auto()  # ranking model
    IMAGE_GENERATION = auto()  # image generation models like DALL-E
    UNDEFINED = auto()


class ModelClientType:
    __doc__ = r"""A quick way to access all model clients in the ModelClient module.

    From this:

    .. code-block:: python

        from adalflow.components.model_client import CohereAPIClient, TransformersClient, AnthropicAPIClient, GroqAPIClient, OpenAIClient

        model_client = OpenAIClient()

    To this:

    .. code-block:: python

        from adalflow.core.types import ModelClientType

        model_client = ModelClientType.OPENAI
    """

    _clients_cache = {}

    def __class_getattr__(cls, name):
        """Dynamically import and return model clients on attribute access."""
        if name in cls._clients_cache:
            return cls._clients_cache[name]

        client_mapping = {
            'COHERE': ('adalflow.components.model_client', 'CohereAPIClient'),
            'TRANSFORMERS': ('adalflow.components.model_client', 'TransformersClient'),
            'ANTHROPIC': ('adalflow.components.model_client', 'AnthropicAPIClient'),
            'GROQ': ('adalflow.components.model_client', 'GroqAPIClient'),
            'OPENAI': ('adalflow.components.model_client', 'OpenAIClient'),
            'GOOGLE_GENAI': ('adalflow.components.model_client', 'GoogleGenAIClient'),
            'OLLAMA': ('adalflow.components.model_client', 'OllamaClient'),
        }

        if name in client_mapping:
            module_name, class_name = client_mapping[name]
            import importlib
            module = importlib.import_module(module_name)
            client_class = getattr(module, class_name)
            cls._clients_cache[name] = client_class
            return client_class

        raise AttributeError(f"'{cls.__name__}' object has no attribute '{name}'")


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
    In sync with OpenAI embedding api spec, same as openai/types/create_embedding_response.py
    """

    prompt_tokens: int
    total_tokens: int


@dataclass
class EmbedderOutput(DataClass):
    __doc__ = r"""Container to hold the response from an Embedder datacomponent for a single batch of input.

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
class CompletionUsage:
    __doc__ = r"In sync with OpenAI completion usage api spec at openai/types/completion_usage.py"
    completion_tokens: Optional[int] = field(
        metadata={"desc": "Number of tokens in the generated completion"}, default=None
    )
    prompt_tokens: Optional[int] = field(
        metadata={"desc": "Number of tokens in the prompt"}, default=None
    )
    total_tokens: Optional[int] = field(
        metadata={
            "desc": "Total number of tokens used in the request (prompt + completion)"
        },
        default=None,
    )


@dataclass
class InputTokensDetails:
    __doc__ = r"Details about input tokens used in a response"
    cached_tokens: Optional[int] = field(
        metadata={"desc": "Number of cached tokens used"}, default=0
    )


@dataclass
class OutputTokensDetails:
    __doc__ = r"Details about output tokens used in a response"
    reasoning_tokens: Optional[int] = field(
        metadata={"desc": "Number of tokens used for reasoning"}, default=0
    )


@dataclass
class ResponseUsage:
    __doc__ = r"Usage information for a response, including token counts, in sync with OpenAI response usage api spec at openai/types/response_usage.py"
    input_tokens: int = field(metadata={"desc": "Number of input tokens used"})
    output_tokens: int = field(metadata={"desc": "Number of output tokens used"})
    total_tokens: int = field(metadata={"desc": "Total number of tokens used"})
    input_tokens_details: InputTokensDetails = field(
        metadata={"desc": "Details about input tokens"},
        default_factory=InputTokensDetails,
    )
    output_tokens_details: OutputTokensDetails = field(
        metadata={"desc": "Details about output tokens"},
        default_factory=OutputTokensDetails,
    )


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

    id: str = field(default=None, metadata={"desc": "The unique id of the output"})

    doc_indices: List[int] = field(
        default=required_field, metadata={"desc": "List of document indices"}
    )
    doc_scores: List[float] = field(
        default=None, metadata={"desc": "List of document scores"}
    )
    query: RetrieverQueryType = field(
        default=None, metadata={"desc": "The query used to retrieve the documents"}
    )
    documents: List[RetrieverDocumentType] = field(
        default=None, metadata={"desc": "List of retrieved documents"}
    )


RetrieverOutputType = Union[
    List[RetrieverOutput], RetrieverOutput
]  # so to support multiple queries at once


#######################################################################################
# Data modeling for function calls
######################################################################################
AsyncCallable = Callable[..., Awaitable[Any]]


@dataclass
class FunctionDefinition(DataClass):
    __doc__ = r"""The data modeling of a function definition, including the name, description, and parameters."""
    # class_instance: Optional[Any] = field(
    #     default=None,
    #     metadata={"desc": "The instance of the class this function belongs to"},
    # )
    # NOTE: for class method: cls_name + "_" + name
    func_name: str = field(
        metadata={"desc": "The name of the tool"}, default=required_field
    )
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
    id: Optional[str] = field(
        default=None, metadata={"desc": "The id of the function call"}
    )
    thought: Optional[str] = field(
        default=None, metadata={"desc": "Your reasoning for this step. Be short for simple queries. For complex queries, provide a clear chain of thought."}
    )  # if the model itself is a thinking model, disable thought field
    name: str = field(default="", metadata={"desc": "The name of the function"})
    args: Optional[List[object]] = field(
        default_factory=list,
        metadata={"desc": "The positional arguments of the function"},
    )
    kwargs: Optional[Dict[str, object]] = field(
        default_factory=dict,
        metadata={"desc": "The keyword arguments of the function"},
    )
    _is_answer_final: Optional[bool] = field(
        default=None,
        metadata={"desc": "Whether this current output is the final answer"},
    )
    _answer: Optional[Any] = field(
        default=None,
        metadata={"desc": "The final answer if this is the final output."},
    )

    @classmethod
    def from_function(
        cls,
        func: Union[Callable[..., Any], AsyncCallable],
        thought: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "Function":
        r"""Create a Function object from a function.

        Args:
            fun (Union[Callable[..., Any], AsyncCallable]): The function to be converted

        Returns:
            Function: The Function object

        Usage:
        1. Create a Function object from a function call:
        2. use :meth:`to_json` and :meth:`to_yaml` to get the schema in JSON or YAML format.
        3. This will be used as an example in prompt showing LLM how to call the function.

        Example:

        .. code-block:: python

            from adalflow.core.types import Function

            def add(a, b):
                return a + b

            # create a function call object with positional arguments
            fun = Function.from_function(add, thought="Add two numbers", 1, 2)
            print(fun)

            # output
            # Function(thought='Add two numbers', name='add', args=[1, 2])
        """
        return cls(
            thought=thought,
            name=func.__name__,
            args=args,
            kwargs=kwargs,
        )

    __output_fields__ = ["thought", "name", "kwargs", "_is_answer_final", "_answer"]


_action_desc = """FuncName(<kwargs>) \
Valid function call expression. \
Example: "FuncName(a=1, b=2)" \
Follow the data type specified in the function parameters.\
e.g. for Type object with x,y properties, use "ObjectType(x=1, y=2)"""


@dataclass
class QueueSentinel:
    """Special sentinel object to mark the end of a stream when using asyncio.Queue for stream processing."""

    type: Literal["queue_sentinel"] = "queue_sentinel"
    """Type discriminator for the sentinel."""


@dataclass
class RawResponsesStreamEvent(DataClass):
    """Streaming event for storing the raw responses from the LLM. These are 'raw' events, i.e. they are directly passed through
    from the LLM.
    """
    input: Optional[Any] = None
    """The input to the LLM."""

    data: Union[Any, None] = None
    """The raw responses streaming event from the LLM."""

    type: Literal["raw_response_event"] = "raw_response_event"
    """The type of the event."""

    error: Optional[str] = None
    """The error message if any."""


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
    id: Optional[str] = field(
        default=None, metadata={"desc": "The unique id of the output"}
    )

    input: Optional[Any] = field(
        default=None,
        metadata={"desc": "The input to the generator"}, # should use it to save the prompt
    )

    data: T_co = field(
        default=None,
        metadata={"desc": "The final output data potentially after output parsers"},
    )  # for reasoning model, this is only the text content/answer (raw_response)
    # extend to support thinking and tool use
    thinking: Optional[str] = field(
        default=None, metadata={"desc": "The thinking of the model"}
    )
    tool_use: Optional[Function] = field(
        default=None, metadata={"desc": "The tool use of the model"}
    )
    images: Optional[Union[str, List[str]]] = field(
        default=None, metadata={"desc": "Generated images (base64 or URLs) from image generation tools"}
    )
    error: Optional[str] = field(
        default=None,
        metadata={"desc": "Error message if any"},
    )
    usage: Optional[CompletionUsage] = field(
        default=None, metadata={"desc": "Usage tracking"}
    )

    # The caller expects the raw_response to follow the OpenAI API documentation for streams
    raw_response: Optional[Union[str, AsyncIterable[T_co], Iterable[T_co]]] = field(
        default=None, metadata={"desc": "Raw string chunk generator from the model"}
    )  # parsed from model client response

    api_response: Optional[Any] = field(
        default=None, metadata={"desc": "Raw response from the api/model client"}
    )
    metadata: Optional[Dict[str, object]] = field(
        default=None, metadata={"desc": "Additional metadata"}
    )

    async def stream_events(self) -> AsyncIterator[T_co]:
        """
        Stream raw events from the Generator's raw response which has the processed version of api_response.
        If the raw_response has already been consumed, yield from the data field

        Returns:
            AsyncIterator[T_co]: An async iterator that yields events stored in raw_response
        """
        count = 0

        # Fallback to raw_response if event_queue didn't yield anything
        if isinstance(self.raw_response, AsyncIterable):
            async for event in self.raw_response:
                count += 1
                yield event

        # if the stream is already consumed and there is final data then just return the final data
        if count == 0 and self.data:
            yield self.data

    def save_images(
        self,
        directory: str = ".",
        prefix: str = "generated",
        format: Literal["png", "jpg", "jpeg", "webp", "gif", "bmp"] = "png",
        decode_base64: bool = True,
        return_paths: bool = True
    ) -> Optional[List[str]]:
        """Save generated images to disk with automatic format conversion.

        Args:
            directory: Directory to save images to (default: current directory)
            prefix: Filename prefix for saved images (default: "generated")
            format: Image format to save as (png, jpg, jpeg, webp, gif, bmp)
            decode_base64: Whether to decode base64 encoded images (default: True)
            return_paths: Whether to return the saved file paths (default: True)

        Returns:
            If return_paths is True:
                - List[str]: Paths to saved images (always returns a list, even for single image)
                - None: If no images to save
            Otherwise returns None

        Examples:
            >>> # Save single image as PNG (returns list with one element)
            >>> response.save_images()
            ['generated_0.png']

            >>> # Save multiple images as JPEG with custom prefix
            >>> response.save_images(prefix="cat", format="jpg")
            ['cat_0.jpg', 'cat_1.jpg']

            >>> # Save to specific directory
            >>> response.save_images(directory="/tmp/images", format="webp")
            ['/tmp/images/generated_0.webp']
        """
        if not self.images:
            return None

        import os
        import base64
        from pathlib import Path

        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)

        saved_paths = []
        images_to_save = self.images if isinstance(self.images, list) else [self.images]

        try:
            # Try to import PIL for format conversion
            from PIL import Image
            import io
            has_pil = True
        except ImportError:
            has_pil = False
            if format.lower() not in ["png", "jpg", "jpeg"]:
                raise ImportError(
                    f"PIL/Pillow is required for '{format}' format. "
                    "Install with: pip install Pillow"
                )

        for i, img_data in enumerate(images_to_save):
            # Determine if this is base64 or a URL
            is_base64 = False
            if isinstance(img_data, str):
                if img_data.startswith("data:"):
                    # Data URI format
                    is_base64 = True
                    # Extract base64 part from data URI
                    base64_data = img_data.split(",")[1] if "," in img_data else img_data
                elif not img_data.startswith(("http://", "https://")):
                    # Assume it's raw base64 if not a URL
                    is_base64 = True
                    base64_data = img_data

            # Construct filename
            filename = f"{prefix}_{i}.{format}"
            filepath = os.path.join(directory, filename)

            if is_base64 and decode_base64:
                # Decode base64 and save
                img_bytes = base64.b64decode(base64_data)

                if has_pil and format.lower() not in ["png"]:
                    # Use PIL to convert format
                    img = Image.open(io.BytesIO(img_bytes))
                    # Convert RGBA to RGB for JPEG
                    if format.lower() in ["jpg", "jpeg"] and img.mode == "RGBA":
                        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                        rgb_img.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                        img = rgb_img
                    # PIL expects 'JPEG' for jpg/jpeg formats
                    pil_format = "JPEG" if format.lower() in ["jpg", "jpeg"] else format.upper()
                    img.save(filepath, pil_format)
                else:
                    # Save as-is (assuming PNG or no conversion needed)
                    with open(filepath, "wb") as f:
                        f.write(img_bytes)
            else:
                # For URLs or if not decoding, save the string as-is
                with open(filepath + ".url", "w") as f:
                    f.write(img_data)
                filepath = filepath + ".url"

            saved_paths.append(filepath)

        if return_paths:
            return saved_paths  # Always return a list
        return None


GeneratorOutputType = GeneratorOutput[object]


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
    # question: str = field(
    #     default=None, metadata={"desc": "The question to ask the LLM"}
    # )
    thought: str = field(default=None, metadata={"desc": "Why the function is called"})
    action: str = field(
        default_factory=required_field,
        metadata={"desc": _action_desc},
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

        Example:

        .. code-block:: python

            from adalflow.core.types import FunctionExpression

            def add(a, b):
                return a + b

            # create an expression for the function call and using keyword arguments
            fun_expr = FunctionExpression.from_function(
                add, thought="Add two numbers", a=1, b=2
            )
            print(fun_expr)

            # output
            # FunctionExpression(thought='Add two numbers', action='add(a=1, b=2)')
        """
        try:
            action = generate_function_call_expression_from_callable(
                func, *args, **kwargs
            )
        except Exception as e:
            logger.error(f"Error generating function expression: {e}")
            raise ValueError(f"Error generating function expression: {e}")
        return cls(action=action, thought=thought)


FunctionOutputValueType = Union[
    Any,
    Generator[Any, Any, Any],
    AsyncGenerator[Any, Any],
    Coroutine[Any, Any, Any],
]


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
    output: Optional[FunctionOutputValueType] = field(
        default=None,
        metadata={
            "desc": "The output of the function execution - supports sync functions, sync generators, async functions, and async generators"
        },
    )
    error: Optional[str] = field(
        default=None, metadata={"desc": "The error message if any"}
    )


#######################################################################################
# Data modeling for component tool
######################################################################################
@dataclass
class ToolOutput(DataClass):
    """Using ToolOutput means a completed tool call even if it has error"""
    output: Any = field(
        default=None, metadata={"description": "The output of the tool"}
    )
    observation: Optional[str] = field(
        default=None,
        metadata={
            "description": "The observation of the llm see of the output of the tool"
        },
    )  # for llm
    display: Optional[str] = field(
        default=None,
        metadata={"description": "The display of the tool output for user"},
    )  # for user
    is_streaming: Optional[bool] = field(
        default=False, metadata={"description": "Whether the tool output is streaming"}
    )
    metadata: Optional[Dict[str, Any]] = field(
        default=None, metadata={"description": "Additional metadata"}
    )
    status: Literal["success", "cancelled", "error"] = field(
        default="success", metadata={"description": "The status of a completed tool call"}
    )


#######################################################################################
# Data modeling for agent component
######################################################################################
@dataclass
class StepOutput(DataClass, Generic[T]):
    __doc__ = r"""The output of a single step in the agent. Suits for serial planning agent such as React"""
    step: int = field(
        default=0, metadata={"desc": "The order of the step in the agent"}
    )

    # This action can be in Function, or Function Exptression, or just str
    # it includes the thought and action already
    # directly the output from planner LLMs
    planner_prompt: Optional[str] = field(
        default=None, metadata={"desc": "The planner prompt for this step"}
    )
    action: T = field(
        default=None, metadata={"desc": "The action the agent takes at this step"}
    )

    function: Optional[Function] = field(
        default=None, metadata={"desc": "The parsed function from the action"}
    )

    observation: Optional[str] = field(
        default=None, metadata={"desc": "The execution result shown for this action"}
    )
    ctx: Optional[Dict[str, Any]] = field(
        default=None, metadata={"desc": "The context of the step"}
    )

    def to_prompt_str(self) -> str:
        output: Dict[str, Any] = {}
        if self.action and isinstance(self.action, FunctionExpression):
            if self.action.thought:
                output["thought"] = self.action.thought
            output["action"] = self.action.action if self.action else None
        if self.observation:
            output["observation"] = (
                self.observation.to_dict()
                if hasattr(self.observation, "to_dict")
                else str(self.observation)
            )
        return json.dumps(output)


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
    metadata: Optional[Dict[str, Any]] = (
        None  # context or files can be used in the user queries
    )


@dataclass
class AssistantResponse:
    response_str: str
    metadata: Optional[Dict[str, Any]] = None  # for agent, we have step history


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
        - Assistant: Doing great!
        DialogTurn(id=uuid4(), user_query=UserQuery("Hi, how are you?"), assistant_response=AssistantResponse("Doing great!"))
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


##############################
# Agent runner events
##############################

import asyncio


@dataclass
class RunItem(DataClass):
    """
    Base class for streaming execution events in the Runner system.

    RunItems represent discrete events that occur during the execution of an Agent
    through the Runner. These items are used for streaming real-time updates about
    the execution progress, allowing consumers to monitor and react to different
    phases of agent execution.

    Attributes:
        id: Unique identifier for tracking this specific event instance
        type: String identifier for the event type (used for event filtering/routing)
        data: Optional generic data payload (deprecated, prefer specific fields in subclasses)
        timestamp: When this event was created (for debugging and monitoring)

    Usage:
        This is an abstract base class. Use specific subclasses for different event types.

    Example:
        ```python
        # Don't instantiate directly - use subclasses
        tool_call_event = ToolCallRunItem(function=my_function)
        ```
    """

    id: str = field(
        default_factory=lambda: str(uuid.uuid4()),
        metadata={"desc": "Unique identifier for this run item"},
    )
    type: str = field(
        default="base",
        metadata={
            "desc": "Type of run item - used for event identification and routing"
        },
    )
    data: Optional[Any] = field(
        default=None,
        metadata={
            "desc": "Generic data payload (deprecated - use specific fields in subclasses)"
        },
    )
    error: Optional[str] = field(
        default=None,
        metadata={"desc": "Error message if an error occurred"},
    )
    timestamp: datetime = field(
        default_factory=datetime.now,
        metadata={"desc": "Timestamp when this event was created"},
    )


@dataclass
class ToolCallRunItem(RunItem):
    """
    Event emitted when the Agent is about to execute a function/tool call.

    This event is generated after the planner LLM has decided on a function to call
    but before the function is actually executed. It allows consumers to monitor
    what tools are being invoked and potentially intervene or log the calls.

    Attributes:
        data: The Function object containing the tool call details (name, args, kwargs)

    Event Flow Position:
        1. Planner generates Function → **ToolCallRunItem** → Function execution → ToolOutputRunItem

    Usage:
        ```python
        # Listen for tool calls in streaming
        async for event in runner.astream(prompt_kwargs).stream_events():
            if isinstance(event, RunItemStreamEvent) and event.name == "tool_called":
                tool_call_item = event.item
                print(f"About to call: {tool_call_item.data.name}")
        ```
    """

    type: str = field(default="tool_call_start", metadata={"desc": "Type of run item"})
    data: Optional[Function] = field(
        default=None,
        metadata={"desc": "Function object containing the tool call to be executed"},
    )


@dataclass
class ToolCallActivityRunItem(RunItem):
    """
    Event emitted during the execution of a function/tool call.

    This event provides intermediate updates on the progress of a function/tool call,
    such as when it starts, completes, or fails. It's paired with ToolCallRunItem to
    provide before/after notification to the caller.

    Attributes:
        data: Any data containing the progress of the tool call

    Event Flow Position:
        ToolCallRunItem → **ToolCallActivityRunItem** → ToolOutputRunItem

    Usage:
        ```python
        # Monitor function execution progress
        async for event in runner.astream(prompt_kwargs).stream_events():
            if isinstance(event, RunItemStreamEvent) and event.name == "tool_call_activity":
                activity_item = event.item
                print(f"Function progress: {activity_item.data}")
        ```
    """

    type: str = field(
        default="tool_call_activity", metadata={"desc": "Type of run item"}
    )
    data: Optional[Any] = field(
        default=None,
        metadata={"desc": "Any data containing the progress of the tool call"},
    )


@dataclass
class FunctionRequest(DataClass):
    """
    Event emitted when the Agent is about to execute a function/tool call.
    """

    id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )  # tool call id for this request
    tool_name: str = field(
        default=None, metadata={"desc": "Name of the tool to be called"}
    )
    tool: Optional[Function] = field(
        default=None,
        metadata={"desc": "Function object containing the tool call to be executed"},
    )
    # send this to the frontend user to display the details of the confirmation
    confirmation_details: Optional[Any] = field(
        default=None,
        metadata={"desc": "Confirmation details for the tool call"},
    )


@dataclass
class ToolCallPermissionRequest(RunItem):
    """
    Event emitted when the Agent is about to execute a function/tool call.

    This event is generated after the planner LLM has decided on a function to call
    but before the function is actually executed. It allows consumers to monitor
    what tools are being invoked and potentially intervene or log the calls.

    Attributes:
        data: The Function object containing the tool call details (name, args, kwargs)

    Event Flow Position:
        1. Planner generates Function → **ToolCallPermissionRequest** → ToolCallPermissionResponse → ToolCallRunItem
        or when user rejects the tool call, ToolCallPermissionRequest → ToolCallPermissionResponse → ToolOutputRunItem

    Usage:
        ```python
        # Listen for tool calls in streaming
        async for event in runner.astream(prompt_kwargs).stream_events():
            if isinstance(event, RunItemStreamEvent) and event.name == "tool_call_permission_request":
                tool_call_item = event.item
                print(f"About to call: {tool_call_item.data.name}")
        ```
    """

    type: str = field(
        default="tool_call_permission_request", metadata={"desc": "Type of run item"}
    )
    data: Optional[FunctionRequest] = field(
        default=None, metadata={"desc": "Function request to be executed"}
    )


@dataclass
class ToolOutputRunItem(RunItem):
    """
    Event emitted after a function/tool call has been executed.

    This event contains the complete execution result, including any outputs or errors
    from the function call. It's paired with ToolCallRunItem to provide before/after
    notification to the caller.

    Attributes:
        data: Complete FunctionOutput containing execution results, errors, etc.

    Event Flow Position:
        ToolCallRunItem → Function execution → **ToolOutputRunItem** → StepRunItem

    Usage:
        ```python
        # Monitor function execution results
        async for event in runner.astream(prompt_kwargs).stream_events():
            if isinstance(event, RunItemStreamEvent) and event.name == "tool_output":
                output_item = event.item
                if output_item.data.error:
                    print(f"Function failed: {output_item.data.error}")
                else:
                    print(f"Function result: {output_item.data.output}")
        ```
    """

    type: str = field(default="tool_output", metadata={"desc": "Type of run item"})
    data: Optional[FunctionOutput] = field(
        default=None,
        metadata={
            "desc": "Complete function execution result including output and error status"
        },
    )


@dataclass
class StepRunItem(RunItem):
    """
    Event emitted when a complete execution step has finished.

    A "step" represents one complete cycle of: planning → tool selection → tool execution.
    This event marks the completion of that cycle and contains the full step information
    including the action taken and the observation (result).

    Attributes:
        data: Complete StepOutput containing step number, action, and observation

    Event Flow Position:
        ToolOutputRunItem → **StepRunItem** → (next step or completion)

    Usage:
        ```python
        # Track step completion
        async for event in runner.astream(prompt_kwargs).stream_events():
            if isinstance(event, RunItemStreamEvent) and event.name == "step_completed":
                step_item = event.item
                print(f"Completed step {step_item.data.step}")
        ```
    """

    type: str = field(default="step", metadata={"desc": "Type of run item"})
    data: Optional[StepOutput] = field(
        default=None,
        metadata={
            "desc": "Complete step execution result including action and observation"
        },
    )

"""
Used to wrap the final response from the runner which holds key information about the execution such
as the answer, step history, and error.
"""


@dataclass
class RunnerResult:
    step_history: List[StepOutput] = field(
        metadata={"desc": "The step history of the execution"},
        default_factory=list,
    )
    answer: Optional[str] = field(
        metadata={"desc": "The answer to the user's query"}, default=None
    )

    error: Optional[str] = field(
        metadata={"desc": "The error message if the code execution failed"},
        default=None,
    )
    ctx: Optional[Dict] = field(
        metadata={"desc": "The context of the execution"},
        default=None,
    )


@dataclass
class FinalOutputItem(RunItem):
    """
    Event emitted when the entire Runner execution has completed.

    This event signals the end of the execution sequence and contains the final
    processed result. It's emitted regardless of whether execution completed
    successfully or with an error.

    Attributes:
        data: The final RunnerResponse containing the complete execution result

    Event Flow Position:
        Final step → **FinalOutputItem** (execution complete)

    Usage:
        ```python
        # Get final results
        async for event in runner.astream(prompt_kwargs).stream_events():
            if isinstance(event, RunItemStreamEvent) and event.name == "runner_finished":
                final_item = event.item
                if final_item.data.error:
                    print(f"Execution failed: {final_item.data.error}")
                else:
                    print(f"Final answer: {final_item.data.answer}")
        ```
    """

    type: str = field(default="final_output", metadata={"desc": "Type of run item"})
    data: Optional[RunnerResult] = field(
        default=None,
        metadata={"desc": "Final processed output from the runner execution"},
    )


@dataclass
class RunItemStreamEvent(DataClass):
    """
    Wrapper for streaming RunItem events during Runner execution.

    This class wraps RunItem instances with event metadata to create a streaming
    event system. Each event has a name that indicates what type of execution
    event occurred, and contains the associated RunItem with the event data.

    The streaming system allows consumers to react to different phases of agent
    execution in real-time, such as when tools are called, when steps complete,
    or when execution finishes.

    Attributes:
        name: The specific event type that occurred (see event name literals)
        item: The RunItem containing the event-specific data
        type: Always "run_item_stream_event" for type discrimination

    Event Types:
        - "agent.final_output": Final output from the agent (FinalOutputItem)
        - "agent.tool_permission_request": Tool permission request before execution
        - "agent.tool_call_start": Agent is about to execute a tool (ToolCallRunItem)
        - "agent.tool_call_activity": Agent is about to execute a tool (ToolCallActivityRunItem)
        - "agent.tool_call_complete": Tool execution completed (ToolOutputRunItem)
        - "agent.step_complete": Full execution step finished (StepRunItem)
        - "agent.execution_complete": Entire execution completed (FinalOutputItem)

    """

    name: Literal[
        # Core agent execution events
        "agent.tool_call_start",  # Function/tool about to be executed
        "agent.tool_call_activity",  # Function/tool intermediate activity and progress updates
        "agent.tool_call_complete",  # Function/tool execution completed
        "agent.step_complete",  # Complete execution step finished
        "agent.final_output",  # Final processed output available
        "agent.execution_complete",  # Entire Runner execution completed
        "agent.tool_permission_request",  # Tool permission request before execution
    ] = field(
        metadata={
            "desc": "The name identifying the specific type of execution event that occurred"
        }
    )
    """The name identifying the specific type of execution event that occurred."""
    # TODO: convert this to data to be consistent with other events
    item: RunItem = field(
        metadata={
            "desc": "The RunItem instance containing the event-specific data and context"
        }
    )
    """The RunItem instance containing the event-specific data and context."""

    type: Literal["run_item_stream_event"] = field(
        default="run_item_stream_event",
        metadata={"desc": "Type discriminator for the streaming event system"},
    )
    """Type discriminator for the streaming event system."""


StreamEvent: TypeAlias = Union[RawResponsesStreamEvent, RunItemStreamEvent]



@dataclass
class QueueCompleteSentinel:
    """Sentinel to indicate queue completion."""

    pass


class EventEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        elif hasattr(obj, "__str__"):
            return str(obj)
        else:
            return super().default(obj)


@dataclass
class RunnerStreamingResult:
    """
    Container for runner streaming results that provides access to the event queue
    and allows users to consume streaming events.
    """

    _event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _run_task: Optional[asyncio.Task] = field(default=None)
    _exception: Optional[Exception] = field(default=None)
    answer: Optional[Any] = field(default=None)
    step_history: List[Any] = field(default_factory=list)
    _is_complete: bool = field(default=False)

    @property
    def is_complete(self) -> bool:
        """Check if the workflow execution is complete."""
        return self._is_complete
    
    def set_exception(self, exc: Any) -> None:
        """Set an exception, ensuring it's a proper exception object."""
        if exc is None:
            self._exception = None
        elif isinstance(exc, BaseException):
            self._exception = exc
        else:
            # Convert non-exception to a proper exception
            self._exception = RuntimeError(f"Non-exception error: {str(exc)}")

    def put_nowait(self, item: StreamEvent):
        # only RawResponsesStreamEvent and RunItemStreamEvent can be put into the queue
        if not isinstance(
            item, (RawResponsesStreamEvent, RunItemStreamEvent, QueueCompleteSentinel)
        ):
            raise ValueError(
                "Only RawResponsesStreamEvent and RunItemStreamEvent can be put into the queue"
            )

        self._event_queue.put_nowait(item)

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        """
        Stream events from the runner execution.w

        Returns:
            AsyncIterator[StreamEvent]: An async iterator that yields stream events

        Example:
            ```python
            result = runner.astream(prompt_kwargs)
            async for event in result.stream_events():
                if isinstance(event, RawResponsesStreamEvent):
                    print(f"Raw event: {event.data}")
                elif isinstance(event, RunItemStreamEvent):
                    print(f"Run item: {event.name} - {event.item}")
            ```
        """
        while True:
            if self._exception:
                # Ensure we're raising a proper exception
                if isinstance(self._exception, BaseException):
                    raise self._exception
                else:
                    # Convert non-exception to a proper exception
                    raise RuntimeError(str(self._exception))

            try:
                # Wait for an event from the queue
                event = await self._event_queue.get()

                # Check for completion sentinel or special completion events
                if isinstance(event, QueueCompleteSentinel):
                    self._event_queue.task_done()
                    break
                else:
                    # always yield event
                    yield event
                    # mark the task as done
                    self._event_queue.task_done()
                    # if the event is a RunItemStreamEvent and the name is agent.execution_complete then additionally break the loop
                    if (
                        isinstance(event, RunItemStreamEvent)
                        and event.name == "agent.execution_complete"
                    ):
                        break

            except asyncio.CancelledError:
                # Clean up and re-raise to allow proper cancellation
                self._is_complete = True
                raise
            except Exception as e:
                # Store unexpected exceptions
                self.set_exception(e)
                raise

    async def stream_to_json(
        self, file_name: str = "agent_events_stream.json"
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream events to a JSON file in real-time while also yielding them.

        This method writes events to a JSON file as they arrive, giving a live
        streaming effect. The JSON file is updated incrementally.

        Args:
            file_name: The output file name for saving events

        Yields:
            StreamEvent: Each event as it arrives

        Example:
            ```python
            result = runner.astream(prompt_kwargs)
            async for event in result.stream_to_json("live_events.json"):
                # Process event while it's also being written to file
                print(f"Event: {event}")
            ```
        """
        event_count = 0

        # Open file and write the opening bracket
        with open(file_name, "w") as f:
            f.write("[\n")

        first_event = True

        try:
            async for event in self.stream_events():
                event_count += 1

                # Prepare event data
                try:
                    if hasattr(event, "to_dict"):
                        event_data = event.to_dict()
                    else:
                        event_data = str(event)
                except Exception as e:
                    # If serialization fails, use a fallback representation
                    event_data = f"<Error serializing event: {str(e)}>"

                event_dict = {
                    "event_number": event_count,
                    "timestamp": datetime.now().isoformat(),
                    "event_type": type(event).__name__,
                    "event_data": event_data,
                }

                # Append to file in streaming fashion
                try:
                    with open(file_name, "r+") as f:
                        # Seek to end of file
                        f.seek(0, 2)

                        if first_event:
                            # For first event, we're right after "[\n"
                            first_event = False
                        else:
                            # For subsequent events, go back to overwrite the previous "\n]"
                            f.seek(f.tell() - 2)
                            f.write(",\n")

                        # Write the event
                        json.dump(event_dict, f, indent=2, cls=EventEncoder)

                        # Write closing bracket
                        f.write("\n]")
                except (IOError, OSError) as e:
                    # Log file write error but continue streaming
                    logger.warning(f"Failed to write event to {file_name}: {e}")

                # Yield the event so caller can process it
                yield event
        except asyncio.CancelledError:
            # Properly handle cancellation
            raise
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Error in stream_to_json: {e}")
            raise

        print(f"\nStreamed {event_count} events to {file_name}")

    def stream_to_json_sync(self, file_name: str = "agent_events_stream.json"):
        """
        Synchronous wrapper for stream_to_json that returns an iterator.

        This allows users to use the streaming JSON functionality in a sync context.

        Args:
            file_name: The output file name for saving events

        Returns:
            Iterator of events

        Example:
            ```python
            result = runner.astream(prompt_kwargs)
            for event in result.stream_to_json_sync("live_events.json"):
                print(f"Event: {event}")
            ```
        """
        import asyncio

        async def _collect_events():
            events = []
            async for event in self.stream_to_json(file_name):
                events.append(event)
            return events

        try:
            # Get the current event loop if running
            # loop = asyncio.get_running_loop()
            # If we're already in an async context, we need to run in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _collect_events())
                return future.result()
        except RuntimeError:
            # No event loop running, we can use asyncio.run directly
            return asyncio.run(_collect_events())

    def cancel(self):
        """Cancel the running task."""
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()

    async def wait_for_completion(self):
        """Wait for the runner task to complete."""
        if self._run_task:
            await self._run_task
