from lightrag.components.model_client.transformers_client import (
    TransformersClient,
)
from lightrag.core.types import ModelType
from lightrag.utils import get_logger

get_logger(level="DEBUG")

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
zephyr = "HuggingFaceH4/zephyr-7b-beta"
context = "Brian is in the kitchen."

rag_prompt_task_desc = {
    "task_desc_str": r"""
You are a helpful assistant.

Your task is to answer the query that may or may not come with context information.
When context is provided, you should stick to the context and less on your prior knowledge to answer the query.

Output JSON format:
{
    "answer": "The answer to the query",
}"""
}
rag_template = r"""<START_OF_SYSTEM_MESSAGE>
You are a helpful assistant.

Your task is to answer the query that may or may not come with context information.
When context is provided, you should stick to the context and less on your prior knowledge to answer the query.
<END_OF_SYSTEM_MESSAGE>
<START_OF_USER_MESSAGE>
    <START_OF_QUERY>
    {{input_str}}
    <END_OF_QUERY>
    {% if context_str %}
    <START_OF_CONTEXT>
    {{context_str}}
    <END_OF_CONTEXT>
    {% endif %}
<END_OF_USER_MESSAGE>
"""


model_kwargs = {
    "model": zephyr,  # "sentence-transformers/all-MiniLM-L6-v2",
    "temperature": 1,
    "stream": False,
}
# transformer_llm = TransformerLLM()
transformer_client = TransformersClient()

kwargs = {
    "model": "HuggingFaceH4/zephyr-7b-beta",
    "temperature": 1,
    "stream": False,
}
query = "where is brian?"

api_kwargs = transformer_client.convert_inputs_to_api_kwargs(
    input=query, model_kwargs=kwargs, model_type=ModelType.LLM
)
print(f"api_kwargs: {api_kwargs}")

expected_api_kwargs = {
    "model": "HuggingFaceH4/zephyr-7b-beta",
    "temperature": 1,
    "stream": False,
    "messages": [{"role": "system", "content": "where is brian?"}],
}

output = transformer_client.call(api_kwargs=api_kwargs, model_type=ModelType.LLM)
print(f"output: {output}")

prompt_kwargs = {
    "input_str": query,
    "context_str": context,
}

# llm_client = CustomLlmModelClient(transformer_llm, auto_model=AutoModelForCausalLM)
# generator = Generator(
#     model_client=transformer_client,
#     model_kwargs=model_kwargs,
#     prompt_kwargs=rag_prompt_task_desc,
#     template=rag_template,
#     # output_processors=JsonParser(),
# )

# output = generator(prompt_kwargs=prompt_kwargs)
