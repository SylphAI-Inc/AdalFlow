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

Here is our manual prompt for the task:

````python

CLASSIFICATION_TASK_DESC = r"""You are a classifier. Given a Question, you need to classify it into one of the following classes:
Format: class_index. class_name, class_description
{% for class in classes %}
{{loop.index-1}}. {{class.label}}, {{class.desc}}
{% endfor %}
"""

TEMPLATE = r"""{# task desc #}
{% if task_desc_str %}
{{task_desc_str}}
{% endif %}
{%if output_format_str %}
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
{% endif %}
{# example #}
{% if examples_str %}
<EXAMPLES>
{#{% for example in examples_str %}#}
{{examples_str}}
{#{% endfor %}#}
</EXAMPLES>
{% endif %}
{{input_label}}: {{input}} {# input_label is the prompt argument #}
Your output:
"""
````

With `output_format_str` and `examples_str` as

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

In this end-to-end demo, we have size of dataset as follows:
- `train`: 500
- `eval`: 6 * 6 = 36 (6 classes, 6 examples per class)
- `test`: 16 * 6 = 96 (16 classes, 6 examples per class)
### Before optimizing

Before we optimize our task pipeline, we will do two evaluations:
1. zero-shot evaluation to see the performance of your manual prompt engineering.
2. Few-shot evaluation where we will check the performance of few-shot ICL either its random, class-balanced, or retrieval-based. This is to see without advanced optimization, just random sample 5 times, how sensitive the task pipeline is to the examples inputed.

Now, lets do this on model `gemma-7b-it`, along with model kwargs as:
```python
 groq_model_kwargs = {
            "model": "gemma-7b-it",  
            "temperature": 0.0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
        }
```
Here is what we message we sent to the model:
```python
{
'json_data': {'messages': [{'role': 'system', 'content': "You are a classifier. Given a Question, you need to classify it into one of the following classes:\nFormat: class_index. class_name, class_description\n0. ABBR, Abbreviation\n1. ENTY, Entity\n2. DESC, Description and abstract concept\n3. HUM, Human being\n4. LOC, Location\n5. NUM, Numeric value\n\n<OUTPUT_FORMAT>\nYour output should be formatted as a standard YAML instance with the following schema:\n```\nthought: Your reasoning to classify the question to class_name (str) (required)\nclass_name: class_name (str) (required)\nclass_index: class_index in range[0, 5] (int) (required)\n```\n\n-Make sure to always enclose the YAML output in triple backticks (```). Please do not add anything other than valid YAML output!\n-Follow the YAML formatting conventions with an indent of 2 spaces. \n-Quote the string values properly.\n\n</OUTPUT_FORMAT>\n<EXAMPLES>\nQuestion: What is SAP ?\nthought: SAP is an abbreviation for a software company and a type of business software. \nclass_name: Abbreviation \nclass_index: 0\n--------\n\nQuestion: What sport is Chris Jogis a top player of ?\nthought: Chris Jogis is known for his achievements in a specific sport, so this question asks about an entity. \nclass_name: Entity \nclass_index: 1\n--------\n\nQuestion: How do you get silly putty out of fabric ?\nthought: The question is asking for a method or explanation of how to remove silly putty from fabric, which fits into the Description and abstract concept category. \nclass_name: Description and abstract concept \nclass_index: 2\n--------\n\nQuestion: Who wrote the Farmer 's Almanac ?\nthought: The question asks for the author, which refers to a human being. \nclass_name: Human being \nclass_index: 3\n--------\n\nQuestion: Where can I get a photograph of professor Randolph Quirk ?\nthought: The question asks for a place where a photograph can be obtained, which pertains to a location. \nclass_name: Location \nclass_index: 4\n--------\n\nQuestion: When was the battle of the Somme fought ?\nthought: The question asks for a specific date or time when the battle occurred, which is a numeric value. \nclass_name: Numeric value \nclass_index: 5\n--------\n\n</EXAMPLES>\nQuestion: What is Ursa Major ? Your output:"}], 'model': 'gemma-7b-it', 'frequency_penalty': 0, 'n': 1, 'presence_penalty': 0, 'temperature': 0.0, 'top_p': 1}
}
```



### Optimizing

Our goals are to improve the performance of 
1. `task_desc_str` via the `LLMOptimizer` with our manual `task_desc_str` as the initial prompt.
2. `few-shot` optimizer should perform better even than the random optimizer.

### After optimizing


## Optimizing it with model finetuning using TorchTune [Optional]
Not necessary as for a classification any LLM is an over-kill.