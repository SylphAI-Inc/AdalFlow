
<!-- <h4 align="center">
    <img alt="AdalFlow logo" src="docs/source/_static/images/adalflow-logo.png" style="width: 100%;">
</h4> -->

<h4 align="center">
    <img alt="AdalFlow logo" src="https://raw.githubusercontent.com/SylphAI-Inc/LightRAG/main/docs/source/_static/images/adalflow-logo.png" style="width: 100%;">
</h4>


<p align="center">
    <a href="https://colab.research.google.com/drive/1TKw_JHE42Z_AWo8UuRYZCO2iuMgyslTZ?usp=sharing">
        <img alt="Try Quickstart in Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="https://adalflow.sylph.ai/">All Documentation</a> |
        <a href="https://adalflow.sylph.ai/apis/components/components.model_client.html">Models</a> |
        <a href="https://adalflow.sylph.ai/apis/components/components.retriever.html">Retrievers</a> |
        <a href="https://adalflow.sylph.ai/apis/components/components.agent.html">Agents</a> |
        <a href="https://adalflow.sylph.ai/use_cases/question_answering.html">Trainer & Optimizers</a>
    <p>
</h4>

<p align="center">
    <a href="https://pypi.org/project/adalflow/">
        <img alt="PyPI Version" src="https://img.shields.io/pypi/v/adalflow?style=flat-square">
    </a>
    <a href="https://star-history.com/#SylphAI-Inc/LightRAG">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/SylphAI-Inc/LightRAG?style=flat-square">
    </a>
    <a href="https://github.com/SylphAI-Inc/LightRAG/issues">
        <img alt="Open Issues" src="https://img.shields.io/github/issues-raw/SylphAI-Inc/LightRAG?style=flat-square">
    </a>
    <a href="https://opensource.org/license/MIT">
        <img alt="License" src="https://img.shields.io/github/license/SylphAI-Inc/LightRAG">
    </a>
      <a href="https://discord.gg/ezzszrRZvT">
        <img alt="discord-invite" src="https://dcbadge.vercel.app/api/server/ezzszrRZvT?style=flat">
    </a>
</p>



<!-- <a href="https://colab.research.google.com/drive/1PPxYEBa6eu__LquGoFFJZkhYgWVYE6kh?usp=sharing">
        <img alt="Try Quickstart in Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a> -->

<!-- <a href="https://pypistats.org/packages/lightrag">
<img alt="PyPI Downloads" src="https://img.shields.io/pypi/dm/lightRAG?style=flat-square">
</a> -->



<h2>
    <p align="center">
     ⚡ The Library to Build and to Auto-optimize LLM Applications ⚡
    </p>
</h2>


AdalFlow helps developers build and optimize LLM task pipelines.
Embracing similar design pattern to PyTorch, AdalFlow is light, modular, and robust, with a 100% readable codebase.


# Why AdalFlow?

LLMs are like water; they can be shaped into anything, from GenAI applications such as chatbots, translation, summarization, code generation, and autonomous agents to classical NLP tasks like text classification and named entity recognition. They interact with the world beyond the model’s internal knowledge via retrievers, memory, and tools (function calls). Each use case is unique in its data, business logic, and user experience.

Because of this, no library can provide out-of-the-box solutions. Users must build towards their own use case. This requires the library to be modular, robust, and have a clean, readable codebase. The only code you should put into production is code you either 100% trust or are 100% clear about how to customize and iterate.

<!-- This is what AdalFlow is: light, modular, and robust, with a 100% readable codebase. -->


Further reading: [How We Started](https://www.linkedin.com/posts/li-yin-ai_both-ai-research-and-engineering-use-pytorch-activity-7189366364694892544-Uk1U?utm_source=share&utm_medium=member_desktop),
[Introduction](https://adalflow.sylph.ai/), [Design Philosophy](https://adalflow.sylph.ai/tutorials/lightrag_design_philosophy.html) and [Class hierarchy](https://adalflow.sylph.ai/tutorials/class_hierarchy.html).


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
# AdalFlow Task Pipeline

We will ask the model to respond with ``explanation`` and ``example`` of a concept. To achieve this, we will build a simple pipeline to get the structured output as ``QAOutput``.

## Well-designed Base Classes

This leverages our two and only powerful base classes: `Component` as building blocks for the pipeline and `DataClass` to ease the data interaction with LLMs.

```python

from dataclasses import dataclass, field

from adalflow.core import Component, Generator, DataClass
from adalflow.components.model_client import GroqAPIClient
from adalflow.components.output_parsers import JsonOutputParser

@dataclass
class QAOutput(DataClass):
    explanation: str = field(
        metadata={"desc": "A brief explanation of the concept in one sentence."}
    )
    example: str = field(metadata={"desc": "An example of the concept in a sentence."})



qa_template = r"""<SYS>
You are a helpful assistant.
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
</SYS>
User: {{input_str}}
You:"""

class QA(Component):
    def __init__(self):
        super().__init__()

        parser = JsonOutputParser(data_class=QAOutput, return_data_class=True)
        self.generator = Generator(
            model_client=GroqAPIClient(),
            model_kwargs={"model": "llama3-8b-8192"},
            template=qa_template,
            prompt_kwargs={"output_format_str": parser.format_instructions()},
            output_processors=parser,
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

## Clear Pipeline Structure

Simply by using `print(qa)`, you can see the pipeline structure, which helps users understand any LLM workflow quickly.

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
      You:, prompt_kwargs: {'output_format_str': 'Your output should be formatted as a standard JSON instance with the following schema:\n```\n{\n    "explanation": "A brief explanation of the concept in one sentence. (str) (required)",\n    "example": "An example of the concept in a sentence. (str) (required)"\n}\n```\n-Make sure to always enclose the JSON output in triple backticks (```). Please do not add anything other than valid JSON output!\n-Use double quotes for the keys and string values.\n-Follow the JSON formatting conventions.'}, prompt_variables: ['output_format_str', 'input_str']
    )
    (model_client): GroqAPIClient()
    (output_processors): JsonOutputParser(
      data_class=QAOutput, examples=None, exclude_fields=None, return_data_class=True
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
  )
)
```

**The Output**

We structure the output to both track the data and potential errors if any part of the Generator component fails.
Here is what we get from ``print(output)``:

```
GeneratorOutput(data=QAOutput(explanation='LLM stands for Large Language Model, which refers to a type of artificial intelligence designed to process and generate human-like language.', example='For instance, LLMs are used in chatbots and virtual assistants, such as Siri and Alexa, to understand and respond to natural language input.'), error=None, usage=None, raw_response='```\n{\n  "explanation": "LLM stands for Large Language Model, which refers to a type of artificial intelligence designed to process and generate human-like language.",\n  "example": "For instance, LLMs are used in chatbots and virtual assistants, such as Siri and Alexa, to understand and respond to natural language input."\n}', metadata=None)
```
**Focus on the Prompt**

Use the following code will let us see the prompt after it is formatted:

```python

qa2.generator.print_prompt(
        output_format_str=qa2.generator.output_processors.format_instructions(),
        input_str="What is LLM?",
)
```


The output will be:

````markdown
<SYS>
You are a helpful assistant.
<OUTPUT_FORMAT>
Your output should be formatted as a standard JSON instance with the following schema:
```
{
    "explanation": "A brief explanation of the concept in one sentence. (str) (required)",
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
````

## Model-agnostic


You can switch to any model simply by using a different `model_client` (provider) and `model_kwargs`.
Let's use OpenAI's `gpt-3.5-turbo` model.

```python
from adalflow.components.model_client import OpenAIClient

self.generator = Generator(
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-3.5-turbo"},
    template=qa_template,
    prompt_kwargs={"output_format_str": parser.format_instructions()},
    output_processors=parser,
)
```


# Quick Install

Install AdalFlow with pip:

```bash
pip install adalflow
```

Please refer to the [full installation guide](https://adalflow.sylph.ai/get_started/installation.html) for more details.




# Documentation

AdalFlow full documentation available at [adalflow.sylph.ai](https://adalflow.sylph.ai/):
- [How We Started](https://www.linkedin.com/posts/li-yin-ai_both-ai-research-and-engineering-use-pytorch-activity-7189366364694892544-Uk1U?utm_source=share&utm_medium=member_desktop)
- [Introduction](https://adalflow.sylph.ai/)
- [Full installation guide](https://adalflow.sylph.ai/get_started/installation.html)
- [Design philosophy](https://adalflow.sylph.ai/tutorials/lightrag_design_philosophy.html)
- [Class hierarchy](https://adalflow.sylph.ai/tutorials/class_hierarchy.html)
- [Tutorials](https://adalflow.sylph.ai/tutorials/index.html)
- [Supported Models](https://adalflow.sylph.ai/apis/components/components.model_client.html)
- [Supported Retrievers](https://adalflow.sylph.ai/apis/components/components.retriever.html)
- [API reference](https://adalflow.sylph.ai/apis/index.html)


# AdalFlow: A Tribute to Ada Lovelace

AdalFlow is named in honor of [Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace), the pioneering female mathematician who first recognized that machines could do more than just calculations. As a female-led team, we aim to inspire more women to enter the AI field.

# Contributors

[![contributors](https://contrib.rocks/image?repo=SylphAI-Inc/LightRAG&max=2000)](https://github.com/SylphAI-Inc/LightRAG/graphs/contributors)

# Citation

```bibtex
@software{Yin2024AdalFlow,
  author = {Li Yin},
  title = {{AdalFlow: The Library for Large Language Model (LLM) Applications}},
  month = {7},
  year = {2024},
  doi = {10.5281/zenodo.12639531},
  url = {https://github.com/SylphAI-Inc/LightRAG}
}
```
