"""
https://arxiv.org/abs/2201.11903

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

from lightrag.light_rag import OpenAIGenerator, Generator
from typing import Any, List
from jinja2 import Template

DEFAULT_CHAIN_OF_THOUGHT_PROMPT = r"""
<START_OF_SYSTEM_PROMPT>
You are a helpful assistant. Let's think step-by-step concisely to answer user's query.
{#few_shot_examples#}
{% if examples %}
Examples:
{% for example in examples %}
{{ example }}
{% endfor %}
{% endif %}
<END_OF_SYSTEM_PROMPT>
User: {{user_query}}
You:
"""


class ChainOfThought:
    name = "Chain of Thought"

    def __init__(
        self,
        generator: Generator = None,
        prompt: str = DEFAULT_CHAIN_OF_THOUGHT_PROMPT,
        examples: List[str] = [],
    ):
        self.generator = generator
        self.prompt = prompt
        self.examples = examples
        self.template = Template(self.prompt)
        # verbose
        print(f"Chain of Thought prompt: {self.prompt}")

    def __call__(self, input: str) -> str:
        prompt = self.template.render(
            user_query=input,
            examples=self.examples,
        )
        messages = [
            {"role": "system", "content": prompt},
        ]
        print(f"messages: {messages}")
        response = self.generator(messages)
        return response


if __name__ == "__main__":
    settings = {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
    }
    generator = OpenAIGenerator(**settings)
    chain_of_thought = ChainOfThought(generator)
    input = "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"
    response = chain_of_thought(input=input)
    print(response)

    # raw response
    messages = [{"role": "system", "content": input}]
    response = generator(messages)
    print(f"raw response: {response}")
