"""
https://arxiv.org/abs/2201.11903, published in Jan, 2023

Chain of the thought(CoT) is to mimic a step-by-step thought process for arriving at the answer. You can achieve it in two ways:
1. Add instructions such as "Let's think step-by-step to answer this question".
2. Add few-shot examples such as
'
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cansof 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
'

NOTE: CoT can be helpful for more complicated task, it also varies from task to task and model to model.
For instance, CoT might already be supported in gpt3.5+ api calls.

Benchmark it with and without CoT to see if it helps.
"""

from typing import Dict, Optional
from core.component import Component
from core.prompt_builder import Prompt
from core.generator import Generator
from core.string_parser import JsonParser
from core.api_client import APIClient
from core.default_prompt_template import DEFAULT_LIGHTRAG_PROMPT


COT_TASK_DESC_STR_BASIC = "You are a helpful assistant. Let's think step-by-step (be concise too) to answer user's query."
# Using triple quotes to include JSON-like structure more cleanly
COT_TASK_DESC_STR_WITH_JSON_OUTPUT = f"""
{COT_TASK_DESC_STR_BASIC} Output JSON format: {{"thought": "<The thought process to answer the query>", "answer": "<The answer to the query>"}}
"""

# ChainOfThought will just be a generator with preset_prompt_kwargs of the task_desc_str = COT_TASK_DESC_STR
# additional you can ask it to generate a json with "thought" and "anwer" keys and use jsonParser


class CoTGenerator(Generator):
    r"""
    CoTGenerator is a subclass of Generator with default task_desc_str preset for Chain of Thought.
    Output will be string.
    It is exactly the same as using a Generator.
    Example:
    ```
    cot = CoTGenerator(model_client=model_client, model_kwargs={"model": model})
    ```
    """

    def __init__(
        self,
        *,
        model_client: APIClient,
        model_kwargs: Optional[Dict] = {},
        template: str = DEFAULT_LIGHTRAG_PROMPT,
        preset_prompt_kwargs={"task_desc_str": COT_TASK_DESC_STR_BASIC},
        # preset_prompt_kwargs: Optional[Dict] = None,
        output_processors: Optional[Component] = None,
    ) -> None:

        super().__init__(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=template,
            preset_prompt_kwargs=preset_prompt_kwargs,
            output_processors=output_processors,
        )


class CoTGeneratorWithJsonOutput(Generator):
    r"""
    CoTGeneratorWithJsonOutput is a subclass of Generator with default task_desc_str preset for Chain of Thought.
    Output will be parsed as JSON with "thought" and "answer" keys.
    Example:
    ```
    cot = CoTGeneratorWithJsonOutput(model_client=model_client, model_kwargs={"model": model})
    ```
    """

    def __init__(
        self,
        *,
        model_client: APIClient,
        model_kwargs: Optional[Dict] = {},
        template: str = DEFAULT_LIGHTRAG_PROMPT,
        preset_prompt_kwargs={"task_desc_str": COT_TASK_DESC_STR_WITH_JSON_OUTPUT},
        output_processors: Optional[Component] = JsonParser(),
    ) -> None:

        super().__init__(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=template,
            preset_prompt_kwargs=preset_prompt_kwargs,
            output_processors=output_processors,
        )


# if __name__ == "__main__":
#     from core.generator import Generator
#     from core.openai_client import OpenAIClient
#     from components.api_client.groq_client import GroqAPIClient
#     from core.string_parser import JsonParser
#     import dotenv

#     dotenv.load_dotenv()
#     model_client = GroqAPIClient()
#     model = "llama3-8b-8192"

#     def test_chain_of_thought_basic():

#         cot = Generator(
#             model_client=model_client,
#             model_kwargs={"model": model},
#             preset_prompt_kwargs={"task_desc_str": COT_TASK_DESC_STR_BASIC},
#         )
#         input = "Li adapted her pet Apple in 2017 when Apple was only 2 months old, now we are at year 2024, how old is Li's pet Apple?"

#         answer = cot(input=input)
#         print(answer)

#     def test_chain_of_thought_with_json_output():

#         cot = Generator(
#             model_client=model_client,
#             model_kwargs={"model": model},
#             preset_prompt_kwargs={"task_desc_str": COT_TASK_DESC_STR_WITH_JSON_OUTPUT},
#             output_processors=JsonParser(),
#         )
#         input = "Li adapted her pet Apple in 2017 when Apple was only 2 months old, now we are at year 2024, how old is Li's pet Apple?"

#         answer = cot(input=input)
#         print(answer)

#     test_chain_of_thought_basic()
#     test_chain_of_thought_with_json_output()
