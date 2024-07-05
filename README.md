![LightRAG Logo](https://raw.githubusercontent.com/SylphAI-Inc/LightRAG/main/docs/source/_static/images/LightRAG-logo-doc.jpeg)

[![release](https://img.shields.io/github/v/release/SylphAI-Inc/LightRAG?sort=semver)](https://github.com/SylphAI-Inc/LightRAG/releases)
[![License](https://img.shields.io/github/license/SylphAI-Inc/LightRAG)](https://opensource.org/license/MIT)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/lightRAG?style=flat-square)](https://pypistats.org/packages/lightRAG)
[![GitHub star chart](https://img.shields.io/github/stars/SylphAI-Inc/LightRAG?style=flat-square)](https://star-history.com/#SylphAI-Inc/LightRAG)
[![Dependency Status](https://img.shields.io/librariesio/github/SylphAI-Inc/LightRAG?style=flat-square)](https://libraries.io/github/SylphAI-Inc/LightRAG)
[![Open Issues](https://img.shields.io/github/issues-raw/SylphAI-Inc/LightRAG?style=flat-square)](https://github.com/SylphAI-Inc/LightRAG/issues)
[![](https://dcbadge.vercel.app/api/server/zt2mTPcu?compact=true&style=flat)](https://discord.gg/zt2mTPcu)


### ⚡ The PyTorch Library for Large language Model (LLM) Applications ⚡

*LightRAG* helps developers with both building and optimizing *Retriever-Agent-Generator (RAG)* pipelines.
It is *light*, *modular*, and *robust*.

<!--

**PyTorch**

```python
import torch
import torch.nn as nn

class Net(nn.Module):
   def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      self.conv2 = nn.Conv2d(32, 64, 3, 1)
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128, 10)

   def forward(self, x):
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.dropout1(x)
      x = self.dropout2(x)
      x = self.fc1(x)
      return self.fc2(x)
``` -->
## LightRAG Task Pipeline


We will ask the model to respond with ``explaination`` and ``example`` of a concept. And we will build a pipeline to get the structured output as ``QAOutput``.

```python

from dataclasses import dataclass, field

from lightrag.core import Component, Generator, DataClass, fun_to_component, Sequential
from lightrag.components.model_client import GroqAPIClient
from lightrag.components.output_parsers import JsonOutputParser

@dataclass
class QAOutput(DataClass):
    explaination: str = field(
        metadata={"desc": "A brief explaination of the concept in one sentence."}
    )
    example: str = field(metadata={"desc": "An example of the concept in a sentence."})


@fun_to_component
def to_qa_output(data: dict) -> QAOutput:
    return QAOutput.from_dict(data)


class QA(Component):
    def __init__(self):
        super().__init__()
        template = r"""<SYS>
You are a helpful assistant.
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
</SYS>
User: {{input_str}}
You:
        """
        parser = JsonOutputParser(data_class=QAOutput)
        self.generator = Generator(
            model_client=GroqAPIClient(),
            model_kwargs={"model": "llama3-8b-8192"},
            template=template,
            prompt_kwargs={"output_format_str": parser.format_instructions()},
            output_processors=Sequential(parser, to_qa_output),
        )

    def call(self, query: str):
        return self.generator.call({"input_str": query})

    async def acall(self, query: str):
        return await self.generator.acall({"input_str": query})
```


Run the following code for visualization and calling the model.

```python

qa = QA()
print(qa)

# call
output = qa("What is LLM?")
print(output)
```

**Structure of the pipeline**

Here is what we get from ``print(qa)``:

```
QA(
  (generator): Generator(
    model_kwargs={'model': 'llama3-8b-8192'},
    (prompt): Prompt(
      template: <SYS>
              You are a helpful assistant.
              <OUTPUT_FORMAT>
              {{output_format_str}}
              </OUTPUT_FORMAT>
              </SYS>
              User: {{input_str}}
              You:
              , prompt_kwargs: {'output_format_str': 'Your output should be formatted as a standard JSON instance with the following schema:\n```\n{\n    "explaination": "A brief explaination of the concept in one sentence. (str) (required)",\n    "example": "An example of the concept in a sentence. (str) (required)"\n}\n```\n-Make sure to always enclose the JSON output in triple backticks (```). Please do not add anything other than valid JSON output!\n-Use double quotes for the keys and string values.\n-Follow the JSON formatting conventions.'}, prompt_variables: ['output_format_str', 'input_str']
    )
    (model_client): GroqAPIClient()
    (output_processors): Sequential(
      (0): JsonOutputParser(
        data_class=QAOutput, examples=None, exclude_fields=None
        (json_output_format_prompt): Prompt(
          template: Your output should be formatted as a standard JSON instance with the following schema:
          ```
          {{schema}}
          ```
          {% if example %}
          Examples:
          ```
          {{example}}
          ```
          {% endif %}
          -Make sure to always enclose the JSON output in triple backticks (```). Please do not add anything other than valid JSON output!
          -Use double quotes for the keys and string values.
          -Follow the JSON formatting conventions., prompt_variables: ['schema', 'example']
        )
        (output_processors): JsonParser()
      )
      (1): ToQaOutputComponent(fun_name=to_qa_output)
    )
  )
)
```

**The output**

Here is what we get from ``print(output)``:

```
GeneratorOutput(data=QAOutput(explaination='LLM stands for Large Language Model, which refers to a type of artificial intelligence designed to process and generate human-like language.', example='For example, a LLM can be trained to generate news articles, conversations, or even entire books, and can be used for a variety of applications such as language translation, text summarization, and chatbots.'), error=None, usage=None, raw_response='```\n{\n  "explaination": "LLM stands for Large Language Model, which refers to a type of artificial intelligence designed to process and generate human-like language.",\n  "example": "For example, a LLM can be trained to generate news articles, conversations, or even entire books, and can be used for a variety of applications such as language translation, text summarization, and chatbots."\n}', metadata=None)
```
**See the prompt**

Use the following code:

```python

qa2.generator.print_prompt(
        output_format_str=qa2.generator.output_processors[0].format_instructions(),
        input_str="What is LLM?",
)
```


The output will be:

```
You are a helpful assistant.
<OUTPUT_FORMAT>
Your output should be formatted as a standard JSON instance with the following schema:
```
{
    "explaination": "A brief explaination of the concept in one sentence. (str) (required)",
    "example": "An example of the concept in a sentence. (str) (required)"
}
```
-Make sure to always enclose the JSON output in triple backticks (```). Please do not add anything other than valid JSON output!
-Use double quotes for the keys and string values.
-Follow the JSON formatting conventions.
</OUTPUT_FORMAT>
</SYS>
User: What is LLM?
You:
```


## Quick Install

Install LightRAG with pip:

```bash
pip install lightrag
```

Please refer to the [full installation guide](https://lightrag.sylph.ai/get_started/installation.html) for more details.




# Documentation

LightRAG full documentation available at [lightrag.sylph.ai](https://lightrag.sylph.ai/):

- [Introduction](https://lightrag.sylph.ai/)
- [Full installation guide](https://lightrag.sylph.ai/get_started/installation.html)
- [Design philosophy](https://lightrag.sylph.ai/developer_notes/lightrag_design_philosophy.html): Design based on three principles: Simplicity over complexity, Quality over quantity, and Optimizing over building.
- [Class hierarchy](https://lightrag.sylph.ai/developer_notes/class_hierarchy.html): We have no more than two levels of subclasses. The bare minimum abstraction provides developers with maximum customizability and simplicity.
- [Tutorials](https://lightrag.sylph.ai/developer_notes/index.html): Learn the `why` and `how-to` (customize and integrate) behind each core part within the `LightRAG` library.
- [API reference](https://lightrag.sylph.ai/apis/index.html)




## Contributors

[![contributors](https://contrib.rocks/image?repo=SylphAI-Inc/LightRAG&max=2000)](https://github.com/SylphAI-Inc/LightRAG/graphs/contributors)

# Citation

```bibtex
@software{Yin2024LightRAG,
  author = {Li Yin},
  title = {{LightRAG: The PyTorch Library for Large Language Model (LLM) Applications}},
  month = {7},
  year = {2024},
  doi = {10.5281/zenodo.12639531},
  url = {https://github.com/SylphAI-Inc/LightRAG}
}
```
