# AdalFlow Optimizer

This directory contains optimization implementations for LLM task pipelines in AdalFlow.

## Recent Updates

### Gumbel-Top-K Sampling and TSGD-M (November 2025)

**New Features:**
- **Gumbel-Top-K Sampling**: Probabilistic prompt selection using Gumbel-Max trick for balanced exploration-exploitation
- **TSGD-M (Textual Gradient Descent with Momentum)**: Enhanced optimizer with momentum-based prompt refinement
- **TrainerGumbel**: Extended trainer class with Gumbel-based selection and multiple history update strategies

**Key Implementations:**

1. **TGDOptimizer** (`text_grad/tgd_optimizer.py`):
   - `gumbel_top_k()`: Implements Gumbel-Top-K sampling with temperature scaling, noise control, and optional UCB bonuses
   - `generate_top_k_scoring_function()`: Softmax acquisition via Gumbel-Max sampling for selecting top-K historical prompts
   - `top_k_selected_prompts()`: Main entry point for Gumbel-based prompt selection during optimization
   - Improved `render_history()`: Better history management with multi-minibatch score tracking

2. **TrainerGumbel** (`trainer/trainer.py`):
   - Extends base Trainer with Gumbel-Top-K selection capabilities
   - **History update strategies**:
     - `improvement_only`: Add to history only when validation score improves
     - `always`: Always add to history regardless of score
     - `epsilon_greedy`: Explore with probability ε, else exploit
     - `confidence_based`: Add if within confidence threshold
   - **TSGD-M workflow**:
     - Maintains prompt cache across iterations
     - Samples K prompts from historical cache based on scores
     - Evaluates sampled prompts on mini-batches before gradient computation
     - Generates next prompt using best historical prompt + gradients
   - **Key methods**:
     - `_fit_text_grad_tsgd_m()`: Main TSGD-M training loop
     - `_sample_prompts_from_cache()`: Sample top-K prompts from cache by score
     - `_evaluate_prompts_on_minibatch()`: Evaluate prompts on validation mini-batch
     - `_add_to_tsgd_m_cache()`: Add prompt, gradient, and scores to cache

**Algorithm Reference:**
- TSGD-M: https://arxiv.org/abs/2506.00400

**Parameters:**
```python
TrainerGumbel(
    # ... base trainer parameters ...
    tsgd_m_enabled=True,              # Enable TSGD-M workflow
    tsgd_m_momentum_window=5,         # K: number of prompts to sample from cache
    history_update_strategy="always", # How to update history
    exploration_epsilon=0.2,          # For epsilon_greedy strategy
    confidence_threshold=0.1          # For confidence_based strategy
)
```

---

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
4. **TSGD-M (Textual Gradient Descent with Momentum)**: Momentum-based optimization that samples and evaluates K prompts from historical cache, then generates new prompts based on best historical prompt + gradients

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
