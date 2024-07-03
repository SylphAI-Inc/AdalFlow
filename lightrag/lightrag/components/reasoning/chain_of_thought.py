"""
Chain of the thought(CoT) is to mimic a step-by-step thought process for arriving at the answer.

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

# from core.component import Component
# from core.generator import Generator
# from core.string_parser import JsonParser
# from core.model_client import ModelClient
# from core.default_prompt_template import DEFAULT_LIGHTRAG_SYSTEM_PROMPT


COT_TASK_DESC_STR_BASIC = (
    "You are a helpful assistant. Let's think step-by-step to answer user's query."
)
# Using triple quotes to include JSON-like structure more cleanly
COT_TASK_DESC_STR_WITH_JSON_OUTPUT = f"""
{COT_TASK_DESC_STR_BASIC} Output JSON format: {{"thought": "<The thought process to answer the query>", "answer": "<The answer to the query>"}}
"""


# ChainOfThought will just be a generator with preset_prompt_kwargs of the task_desc_str = COT_TASK_DESC_STR
# additional you can ask it to generate a json with "thought" and "anwer" keys and use jsonParser


# class CoTGenerator(Generator):
#     r"""
#     CoTGenerator is a subclass of Generator with default task_desc_str preset for Chain of Thought.
#     Output will be string.
#     It is exactly the same as using a Generator.
#     Example:
#     ```
#     cot = CoTGenerator(model_client=model_client, model_kwargs={"model": model})
#     ```
#     """

#     def __init__(
#         self,
#         *,
#         model_client: ModelClient,
#         model_kwargs: Dict = {},
#         template: Optional[str] = None,
#         preset_prompt_kwargs: Optional[Dict] = None,
#         output_processors: Optional[Component] = None,
#     ) -> None:

#         super().__init__(
#             model_client=model_client,
#             model_kwargs=model_kwargs,
#             template=template or DEFAULT_LIGHTRAG_SYSTEM_PROMPT,
#             preset_prompt_kwargs=preset_prompt_kwargs
#             or {"task_desc_str": COT_TASK_DESC_STR_BASIC},
#             output_processors=output_processors,
#         )


# class CoTGeneratorWithJsonOutput(Generator):
#     r"""
#     CoTGeneratorWithJsonOutput is a subclass of Generator with default task_desc_str preset for Chain of Thought.
#     Output will be parsed as JSON with "thought" and "answer" keys.
#     Example:
#     ```
#     cot = CoTGeneratorWithJsonOutput(model_client=model_client, model_kwargs={"model": model})
#     ```
#     """

#     def __init__(
#         self,
#         *,
#         model_client: ModelClient,
#         model_kwargs: Dict = {},
#         template: Optional[str] = None,
#         preset_prompt_kwargs: Optional[Dict] = None,
#         output_processors: Optional[Component] = None,
#     ) -> None:

#         super().__init__(
#             model_client=model_client,
#             model_kwargs=model_kwargs,
#             template=template or DEFAULT_LIGHTRAG_SYSTEM_PROMPT,
#             preset_prompt_kwargs=preset_prompt_kwargs
#             or {"task_desc_str": COT_TASK_DESC_STR_WITH_JSON_OUTPUT},
#             output_processors=output_processors or JsonParser(),
#         )
