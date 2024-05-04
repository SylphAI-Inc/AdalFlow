from typing import Any, Dict, List, Optional, overload
from core.component import Component
from openai.types.chat import ChatCompletion
from core.data_classes import EmbedderOutput
from core.prompt_builder import Prompt

from enum import Enum, auto


class ModelType(Enum):
    EMBEDDER = auto()
    LLM = auto()


class Model(Component):
    r"""
    Similar to nn.Linear, nn.Conv2d, nn.LSTM, etc. in PyTorch. if you want to use something out of the box, you can use this.
    Base class for most Model inference (or potentially training). If your Model does not fit this pattern, you can extend Component directly.
    Either local or via API calls.
    Support Embedder, LLM Generator.
    NOTE: model has no state/memory. It is stateless. It is just a function that takes input and returns output.
    TODO: not finalized yet. This in the future can be separated into Embedder, LLM, ImageModel, etc. if it gets too big.
    """

    # TODO: allow for specifying the model type, e.g. "embedder", "LLM",
    # TODO: support image models
    type: Optional[ModelType]
    prompt: Optional[Prompt]

    def __init__(
        self, provider: str, type: Optional[ModelType] = None, **model_kwargs
    ) -> None:
        super().__init__(provider=provider)
        if "model" not in model_kwargs:
            raise ValueError(
                f"{type(self).__name__} requires a 'model' to be passed in the model_kwargs"
            )
        self.type = type
        self.model_kwargs = model_kwargs
        self.__setattr__("prompt", None)

    # def set_lm_attributes(self, prompt: Optional[Prompt] = None) -> None:
    #     """
    #     Set the attributes of the language model.
    #     """
    #     self.prompt: Optional[Prompt] = prompt

    # define two types, one for embeddings, and one for generator with completions
    @overload
    def call(self, input: str, **model_kwargs) -> EmbedderOutput: ...

    @overload
    def call(self, input: List[str], **model_kwargs) -> EmbedderOutput: ...

    @overload
    def call(self, input: List[Dict], **model_kwargs) -> Any: ...

    """
    Used by chatable generator with list of messages
    """

    @overload
    def call(self, input: str, **model_kwargs) -> str: ...

    """
    Used by simple non-chatable generator
    """

    def call(self, input: Any, **model_kwargs) -> Any:
        raise NotImplementedError(
            f"Model {type(self).__name__} is missing the required 'call' method."
        )

    def _componse_lm_input_chat(self, **kwargs: Any) -> List[Dict]:
        """
        This combines the default lm input using Prompt, and the passed input. history, steps, etc.
        It builds the final chat input to the model.


        """
        # ensure self.prompt is set
        if not hasattr(self, "prompt") or not self.prompt:
            raise ValueError(
                f"{type(self).__name__} requires a 'prompt' to be set before calling the model."
            )
        current_role: str = kwargs.get("role", "system")
        previous_messages: List[Dict] = kwargs.get("previous_messages", [])
        prompt_text = self.prompt.call(**kwargs)
        # llm input or the api's input
        messages = previous_messages + [{"role": current_role, "content": prompt_text}]
        return messages

    def _componse_lm_input_non_chat(self, **kwargs: Any) -> str:
        """
        This combines the default lm input using Prompt, and the passed input. history, steps, etc.
        It builds the final chat input to the model.

        As
        """
        prompt_text = self.prompt.call(**kwargs)
        return prompt_text

    def compose_model_input(self, **kwargs: Any) -> Any:
        """
        Compose the input to the model. It can be a string, list of strings, list of dictionaries, etc.
        """
        # show not implemented error
        raise NotImplementedError(
            f"Model {type(self).__name__} is missing the required 'compose_model_input' method."
        )

    def compose_model_kwargs(self, **model_kwargs) -> Dict:
        r"""
        The model configuration exclude the input itself.
        Combine the default model, model_kwargs with the passed model_kwargs.
        Example:
        model_kwargs = {"temperature": 0.5, "model": "gpt-3.5-turbo"}
        self.model_kwargs = {"model": "gpt-3.5"}
        combine_kwargs(model_kwargs) => {"temperature": 0.5, "model": "gpt-3.5-turbo"}

        """
        pass_model_kwargs = self.model_kwargs.copy()

        if model_kwargs:
            pass_model_kwargs.update(model_kwargs)
        return pass_model_kwargs

    def parse_completion(self, completion: ChatCompletion) -> str:
        """
        Parse the completion to a structure your sytem standarizes. (here is str)
        """
        return completion.choices[0].message.content
