

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

# Loss function

Previously loss function is to get the distance between the generated output and the target output.
the goal is to minimize as it is differentiable. if accuracy is diffentiable (eval metrics), we would not need to use loss function.

In the case of icl, loss function = collect feedback from batch_run. it will be just text. Everyone can use
(1) llm as a loss function
(2) directly convert accuracy -> text feedback.

## GradComponent

Any class inherite from ``GradComponent`` can be thought of a layer in text-grad that can be backpropagated. It will have a forward and backward function. It can be also be used as a loss function.

# The role of a component (note)

In PyTorch, they building blocks will be part of the models. So it is clear each of them will have weights (parameters) and be used with auto-grad to optimize the model.

In LLM applications, component will be used (1) layers to optimize with the usage of its parameters (2) used for its seralization and visualization ability.

We could potentially split it into two levels of inheritance.

(1) [BaseComponent] component without parameters as a way in adalflow to visualize the whole pipeline. even the optimizer as most of them are using llm. This basecomponent will not have to be related to parameters at all.
(2) [Component] component with parameters (module),training as a building block to the actual task pipeline. => name it "Module". To make it able to use text-gradient, it will also be marked as a ``GradComponent``.

# Problem of Text-grad now

1. users have to debug and have control of the prompt themselves. -> We need to make it possible for developers to even debug and customize their optimizer prompt.  [logging + visualization]
2. The gradients should reflect the actual loss. TODO: we will improve
3. The prompts are too complicated and not easy to follow. TODO: simplify the prompt and further improve the data structure.
4. it needs larger batch size so that we can have errors (the actual loss) to backpropagate.

## What really matters to an optimizer

1. use minor changes instead of big changes (we can control the momenetum)
2. The feedback/gradient just looking at a single example is not good for prompt optimization, maybe the instance.

We will differentiate the prompt or instance.


We will have to come with a simplified and more effective version of text-grad, but the structure with our component is solid and powerful.

# Differ from model optimization
1. we have to manually define parameters but in pytorch, it comes inside of a layer, so you just define a layer instead.
