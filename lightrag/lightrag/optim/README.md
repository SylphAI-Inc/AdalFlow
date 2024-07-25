

# Optimization goal

```
class OptimizeGoal(Enum):
    # 1. Similar to normal model auto-grad
    LLM_SYS_INSTRUCTION = auto()  # fixed system prompt instruction across all calls
    LLM_SYS_EXAMPLE = (
        auto()
    )  # few-shot examples (fixed across all calls) => vs dynamic examples
    DYNAMIC_FEW_SHOT_EXAMPLES = auto()  # dynamic examples leverage retrieval
    LLM_RESPONSE = (
        auto()
    )  # similar to reflection, to optimize the response with optimizer
```

# Different optimization strategy

1. orpo: propose, test, and refine, and then test again
2. text-grad: make it the same as the normal model auto-grad, might potentially support multiple parameters/generators
3. Lazy brute-force: run all samples on the pipeline, throw each prompt, examples, input/output to a large-context-window llm and ask it to optimize each part, sys_few_shot.
