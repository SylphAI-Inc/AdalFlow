This is to show how LightRAG is used to optimize a task end to end, from the using of datasets, the configuration,s the setting up of evalutor, the training pipeline on top of the `task pipeline ("Model")` itself.

Besides the auto optimzing of the task pipeline, we also show that how to use this optimized task pipeline to label more training data. And then we train a smaller classifier using embeddings + a classifier head (linear from pytorch or sklean) and train the classifier on the new labeled data.

We compare (1) classifier + llm-synthetic data, (2) classifier + ground truth data, (3) classifier + llm-synthetic data + ground truth data.

And finally you will have a classifier, cheaper and faster to run and perform the same or even better than the original llm task pipeline.
## Task pipeline(Model)
`task.py` along with `config`

In class `TrecClassifier`'s `call` method. Beside of the standard output processing such as `YAMLOutputParser`, we see we add additional **task-specific processing** in case the llm is not following the standard output format (which should be failed predictions).


### Config [Optional]
###  Debugging

1. save the structure of the model (`print(task)`)
2. turn on the library logging 

```python
from utils import enable_library_logging, get_logger


### Prompt Template
## Manual prompt engineering [ICL]
## Auto promot engineering [ICL]
`train.py` is where we do APE for the In-context-learning.

We wrap the zero-shot and few-shot eval in a ICL trainer.

An `ICLTrainer` consists of four parts:
1. `task pipeline` along with task configs.
2. `datasets`, normally `train`, `eval` and `test`.
The `train` is used for providing signals for the optimizer on how to update the generator parameters next. When it is few-shot ICL, the examples in the `train` dataset will be sampled as the `examples_str` in the prompt.
The `eval`/`validation` is for picking the final models, checking early stopping, etc.
The `test` is for accessing the performance of the task pipeline in practice.
3. `optimizer`, which is the optimizer for the generator. It can be any optimizer that is compatible with the `task pipeline`.
4. `evaluator`, which is the evaluator for the task pipeline. It is task-specific.

An `ICLTrainer` itself is highly task-specific too. Our library just provides some basic building blocks and examples to help you build your own `ICLTrainer`.

### Before optimizing

Before we optimize our task pipeline, we will do two evaluations:
1. zero-shot evaluation to see the performance of your manual prompt engineering.
2. Few-shot evaluation where we will check the performance of few-shot ICL either its random, class-balanced, or retrieval-based. This is to see without advanced optimization, just random sample 5 times, how sensitive the task pipeline is to the examples inputed.

### Optimizing

Our goals are to improve the performance of 
1. `task_desc_str` via the `LLMOptimizer` with our manual `task_desc_str` as the initial prompt.
2. `few-shot` optimizer should perform better even than the random optimizer.

### After optimizing


## Optimizing it with model finetuning using TorchTune [Optional]
Not necessary as for a classification any LLM is an over-kill.