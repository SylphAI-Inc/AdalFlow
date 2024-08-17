from typing import Any, Dict, Optional
import logging

import torch

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from adalflow.core.model_client import ModelClient

log = logging.getLogger(__name__)


class TransformerLLM:
    models: Dict[str, type] = {}

    def __init__(self, model_name: Optional[str] = None):
        super().__init__()
        if model_name is not None:
            self.model_name = model_name
            """Lazy intialisation of the model in TransformerClient.init_sync_client()"""

    def init_model(
        self,
        model_name: str,
        auto_model: Optional[type] = AutoModel,
        auto_tokenizer: Optional[type] = AutoTokenizer,
    ):
        try:
            self.tokenizer = auto_tokenizer.from_pretrained(model_name)
            self.model = auto_model.from_pretrained(model_name, is_decoder=True)
            # register the model
            self.models[model_name] = self.model
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"Done loading model {model_name}")
            # Set pad token if it's not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token  # common fallback
                self.model.config.pad_token_id = (
                    self.tokenizer.eos_token_id
                )  # ensure consistency in the model config
        except Exception as e:
            log.error(f"Error loading model {model_name}: {e}")
            raise e

    def parse_chat_completion(self, input_text: str, response: str):
        print("|" * 24)
        print("input lowered")
        print("|" * 24)
        print(input_text.lower().replace("\n", ""))
        print("|" * 24)
        print("|" * 24)
        print("response lowered")
        print("|" * 24)
        print(response.lower().replace("\n", ""))
        print("|" * 24)
        if input_text.lower() in response.lower():
            parsed_response = response.replace(
                input_text, ""
            ).strip()  # Safely handle cases where input_text might not be in response
        else:
            parsed_response = response
        if "xxxxx" in parsed_response:
            cut_idx = parsed_response.find("xxxxx")
            parsed_response = parsed_response[cut_idx + 5 :]
        print(parsed_response)
        return parsed_response

    def call(
        self,
        input_text: str,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
        max_length: int = 150,
    ):
        if not self.model:
            log.error("Model is not initialized.")
            raise ValueError("Model is not initialized.")

        # Ensure tokenizer has pad token; set it if not
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = (
                self.tokenizer.eos_token_id
            )  # Sync model config pad token id

        # Process inputs with attention mask and padding
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(
            self.device
        )

        with torch.no_grad():  # Ensures no gradients are calculated to save memory and computations
            generate_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,  # Control the output length more precisely
                repetition_penalty=5.0,
            )
        response = self.tokenizer.decode(
            generate_ids[0],
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        print("o" * 24)
        print("raw LLM output")
        print("o" * 24)
        print(response)
        print("o" * 24)

        print("o" * 24)
        print("input text")
        print("o" * 24)
        print(input_text)
        print("o" * 24)
        parsed_response = self.parse_chat_completion(
            input_text=input_text, response=response
        )
        return parsed_response

    def __call__(
        self,
        input_text: str,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
        max_length: int = 150,
        model=None,  # For compantibility with Generator ||||| might be something to fix in Generator source code -> api_kwargs always contains a 'model' arguments. 'model' is parsed either in call() or in __init__(),
        **kwargs,
    ):

        return self.call(
            input_text=input_text,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            max_length=max_length,
        )


class CustomLlmModelClient(ModelClient):

    def __init__(
        self,
        llm_model: TransformerLLM,
        auto_model: Optional[type] = AutoModel,
        auto_tokenizer: Optional[type] = AutoTokenizer,
    ) -> None:
        super().__init__()
        self.transformer_llm = llm_model
        self.llm_client = self.init_llm_client(
            auto_model=auto_model, auto_tokenizer=auto_tokenizer
        )

    def init_llm_client(
        self,
        auto_model: Optional[type] = AutoModel,
        auto_tokenizer: Optional[type] = AutoTokenizer,
    ):
        model_name = self.transformer_llm.model_name
        self.transformer_llm.init_model(
            model_name, auto_model=auto_model, auto_tokenizer=auto_tokenizer
        )
        """The transformerLlm is initialised by the user so I removed the parenthesis from the return statement to avoid executing self.transformer_llm.call()"""
        return self.transformer_llm

    def call(
        self, api_kwargs: Dict = {}, model_type=None  # For compatibility with Generator
    ):
        if "model" not in api_kwargs:
            raise ValueError("model must be specified in api_kwargs")
        if not hasattr(self, "llm_client") or self.llm_client is None:
            self.llm_client = self.init_llm_client()
        response = self.llm_client(**api_kwargs)
        return response

    def convert_inputs_to_api_kwargs(
        self,
        input: Any,  # for retriever, it is a single query,
        model_kwargs: dict = {},
        model_type=None,  # For compantibility with Generator
    ) -> dict:
        final_model_kwargs = model_kwargs.copy()
        assert "model" in final_model_kwargs, "model must be specified"
        final_model_kwargs["input_text"] = input
        return final_model_kwargs

    def parse_chat_completion(self, completion: str):
        """Method implemented for compatibility with Generator.
        Return the input of the function without changing it.
        """
        return completion


if __name__ == "__main__":
    from adalflow.core import Generator

    MODEL = "BAAI/bge-small-en-v1.5"
    context = "Brian is in the kitchen."
    query = "where is brian?"

    rag_prompt_task_desc = {
        "task_desc_str": r"""
    You are a helpful assistant.

    Your task is to answer the query that may or may not come with context information.
    When context is provided, you should stick to the context and less on your prior knowledge to answer the query.

    Insert your answer to the query.

    xxxxx
    """
    }
    template = """
            <start_of_system_prompt>
            {# task desc #}
            {% if task_desc_str %}
                {{task_desc_str}}
            {% else %}
                You are a helpful assistant.
            {% endif %}

            {# output format #}
            {% if output_format_str %}
                <OUTPUT_FORMAT>
                    {{output_format_str}}
                </OUTPUT_FORMAT>
            {% endif %}

            {# tools #}
            {% if tools_str %}
                <TOOLS>
                    {{tools_str}}
                </TOOLS>
            {% endif %}

            {# example #}
            {% if examples_str %}
                <EXAMPLES>
                    {{examples_str}}
                </EXAMPLES>
            {% endif %}

            {# chat history #}
            {% if chat_history_str %}
                <CHAT_HISTORY>
                    {{chat_history_str}}
                </CHAT_HISTORY>
            {% endif %}

            {# context #}
            {% if context_str %}
                <CONTEXT>
                    {{context_str}}
                </CONTEXT>
            {% endif %}

            {# steps #}
            {% if steps_str %}
                <STEPS>
                    {{steps_str}}
                </STEPS>
            {% endif %}

            </end_of_system_prompt>

            {% if input_str %}
                <User>
                    {{input_str}}
                </User>
            {% endif %}
            You:


            """

    prompt_kwargs = {
        "input_str": query,
        "context_str": context,
    }
    model_kwargs = {"model": MODEL, "temperature": 1, "stream": False}
    transformer_llm = TransformerLLM(MODEL)
    llm_client = CustomLlmModelClient(transformer_llm, auto_model=AutoModelForCausalLM)
    generator = Generator(
        template=template,
        model_client=llm_client,
        model_kwargs=model_kwargs,
        prompt_kwargs=rag_prompt_task_desc,
        # output_processors=JsonParser()
    )
    print("-" * 24)
    response = generator(prompt_kwargs)
    print(response)
    print("-" * 24)
