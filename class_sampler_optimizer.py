DatasetDict({
    train: Dataset({
        features: ['text', 'coarse_label', 'fine_label'],
        num_rows: 5452
    })
    test: Dataset({
        features: ['text', 'coarse_label', 'fine_label'],
        num_rows: 500
    })
})
Train example: {'text': 'How did serfdom develop in and then leave Russia ?', 'coarse_label': 2, 'fine_label': 26}
Test example: {'text': 'How far is it from Denver to Aspen ?', 'coarse_label': 5, 'fine_label': 40}
INFO:core.prompt_builder:Prompt has variables: ['classes']
INFO:core.prompt_builder:Prompt has variables: ['example', 'schema']
DEBUG:use_cases.classification.task:output_str: Your output should be formatted as a standard YAML instance with the following schema:
```
thought: Your reasoning to classify the question to class_name (str) (required)
class_name: class_name (str) (required)
class_index: class_index in range[0, 5] (int) (required)
```

-Make sure to always enclose the YAML output in triple backticks (```). Please do not add anything other than valid YAML output!
-Follow the YAML formatting conventions with an indent of 2 spaces. 
-Quote the string values properly.

DEBUG:httpx:load_ssl_context verify=True cert=None trust_env=True http2=False
DEBUG:httpx:load_verify_locations cafile='/Users/liyin/Documents/test/LightRAG/.venv/lib/python3.11/site-packages/certifi/cacert.pem'
INFO:core.prompt_builder:Prompt has variables: ['input', 'output_format_str', 'examples_str', 'input_label', 'task_desc_str']
data: None, requires_opt: True
data: {'examples_str': Parameter: None with key: examples_str}, requires_opt: True
Registered parameter trainable_prompt_kwargs with value Parameter: {'examples_str': Parameter: None with key: examples_str} with key: None
