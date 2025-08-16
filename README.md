
<!-- <h4 align="center">
    <img alt="AdalFlow logo" src="docs/source/_static/images/adalflow-logo.png" style="width: 100%;">
</h4> -->



<h4 align="center">
    <img alt="AdalFlow logo" src="https://raw.githubusercontent.com/SylphAI-Inc/AdalFlow/main/docs/source/_static/images/adalflow-logo.png" style="width: 100%;">
</h4>

<h2>
    <p align="center">
     ⚡ AdalFlow is a PyTorch-like library to build and auto-optimize any LM workflows, from Chatbots, RAG,  to Agents. ⚡
    </p>
</h2>




<p align="center">
    <a href="https://colab.research.google.com/drive/1_YnD4HshzPRARvishoU4IA-qQuX9jHrT?usp=sharing">
        <img alt="Try Quickstart in Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="https://adalflow.sylph.ai/">View Documentation</a>
        <!-- <a href="https://adalflow.sylph.ai/apis/components/components.model_client.html">Models</a> |
        <a href="https://adalflow.sylph.ai/apis/components/components.retriever.html">Retrievers</a> |
        <a href="https://adalflow.sylph.ai/apis/components/components.agent.html">Agents</a> |
        <a href="https://adalflow.sylph.ai/tutorials/evaluation.html"> LLM evaluation</a> |
        <a href="https://adalflow.sylph.ai/use_cases/question_answering.html">Trainer & Optimizers</a> -->
    <p>
</h4>
<p align="center">
    <a href="https://pypi.org/project/adalflow/">
        <img alt="PyPI Version" src="https://img.shields.io/pypi/v/adalflow?style=flat-square">
    </a>
    <a href="https://pypi.org/project/adalflow/">
        <img alt="PyPI Downloads" src="https://static.pepy.tech/badge/adalflow">
    </a>
    <a href="https://pypi.org/project/adalflow/">
        <img alt="PyPI Downloads" src="https://static.pepy.tech/badge/adalflow/month">
    </a>
    <a href="https://star-history.com/#SylphAI-Inc/AdalFlow">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/SylphAI-Inc/AdalFlow?style=flat-square">
    </a>
    <a href="https://github.com/SylphAI-Inc/AdalFlow/issues">
        <img alt="Open Issues" src="https://img.shields.io/github/issues-raw/SylphAI-Inc/AdalFlow?style=flat-square">
    </a>
    <a href="https://opensource.org/license/MIT">
        <img alt="License" src="https://img.shields.io/github/license/SylphAI-Inc/AdalFlow">
    </a>
      <a href="https://discord.gg/ezzszrRZvT">
        <img alt="discord-invite" src="https://dcbadge.limes.pink/api/server/ezzszrRZvT?style=flat">
    </a>
</p>

<!-- <h4>
<p align="center">
For AI researchers, product teams, and software engineers who want to learn the AI way.
</p>
</h4> -->

<!-- <h4>
<p align="center">
AdalFlow is a PyTorch-like library to build and auto-optimize any LM workflows, from Chatbots, RAG,  to Agents.
</p>
</h4> -->




<!-- <a href="https://colab.research.google.com/drive/1PPxYEBa6eu__LquGoFFJZkhYgWVYE6kh?usp=sharing">
        <img alt="Try Quickstart in Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a> -->

<!-- <a href="https://pypistats.org/packages/lightrag">
<img alt="PyPI Downloads" src="https://img.shields.io/pypi/dm/lightRAG?style=flat-square">
</a> -->

# Why AdalFlow

1. **100% Open-source Agents SDK**: Lightweight and requires no additional API to setup ``Human-in-the-Loop`` and ``Tracing`` Functionalities.
2. **Say goodbye to manual prompting**: AdalFlow provides a unified auto-differentiative framework for both zero-shot optimization and few-shot prompt optimization. Our research, ``LLM-AutoDiff`` and ``Learn-to-Reason Few-shot In Context Learning``, achieve the highest accuracy among all auto-prompt optimization libraries.
3. **Switch your LLM app to any model via a config**:  AdalFlow provides `Model-agnostic` building blocks for LLM task pipelines, ranging from RAG, Agents to classical NLP tasks.

<!-- <p align="center" style="background-color: #f0f0f0;">
  <img src="https://raw.githubusercontent.com/SylphAI-Inc/AdalFlow/main/docs/source/_static/images/classification_training_map.png" style="width: 80%;" alt="AdalFlow Auto-optimization">
</p> -->

<p align="center" style="background-color: #f0f0f0;">
  <img src="https://raw.githubusercontent.com/SylphAI-Inc/AdalFlow/main/docs/source/_static/images/classification_opt_prompt.png" alt="AdalFlow Optimized Prompt" style="width: 80%;">
</p>

<p align="center" style="background-color: #f0f0f0;">
  <img src="https://raw.githubusercontent.com/SylphAI-Inc/AdalFlow/main/docs/source/_static/images/adalflow_tracing_mlflow.png" alt="AdalFlow MLflow Integration" style="width: 80%;">
</p>

<!-- Among all libraries, AdalFlow achieved the highest accuracy with manual prompting (starting at 82%) and the highest accuracy after optimization. -->
<!-- <p align="center" style="background-color: #f0f0f0;">
  <img src="https://raw.githubusercontent.com/SylphAI-Inc/AdalFlow/main/docs/source/_static/images/classification_opt_prompt.png" alt="AdalFlow Optimized Prompt" style="width: 80%;">
</p> -->

View [Documentation](https://adalflow.sylph.ai)


# Quick Start


Install AdalFlow with pip:

```bash
pip install adalflow
```

## Hello World Agent Example

```python
from adalflow import Agent, Runner
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.core.types import (
    ToolCallActivityRunItem, 
    RunItemStreamEvent,
    ToolCallRunItem,
    ToolOutputRunItem,
    FinalOutputItem
)
import asyncio

# Define tools
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error: {e}"

async def web_search(query: str="what is the weather in SF today?") -> str:
    """Web search on query."""
    await asyncio.sleep(0.5)
    return "San Francisco will be mostly cloudy today with some afternoon sun, reaching about 67 °F (20 °C)."

def counter(limit: int):
    """A counter that counts up to a limit."""
    final_output = []
    for i in range(1, limit + 1):
        stream_item = f"Count: {i}/{limit}"
        final_output.append(stream_item)
        yield ToolCallActivityRunItem(data=stream_item)
    yield final_output

# Create agent with tools
agent = Agent(
    name="MyAgent",
    tools=[calculator, web_search, counter],
    model_client=OpenAIClient(),
    model_kwargs={"model": "gpt-4o", "temperature": 0.3},
    max_steps=5
)

runner = Runner(agent=agent)
```

### 1. Synchronous Call Mode

```python
# Sync call - returns RunnerResult with complete execution history
result = runner.call(
    prompt_kwargs={"input_str": "Calculate 15 * 7 + 23 and count to 5"}
)

print(result.answer)
# Output: The result of 15 * 7 + 23 is 128. The counter counted up to 5: 1, 2, 3, 4, 5.

# Access step history
for step in result.step_history:
    print(f"Step {step.step}: {step.function.name} -> {step.observation}")
# Output:
# Step 0: calculator -> The result of 15 * 7 + 23 is 128
# Step 1: counter -> ['Count: 1/5', 'Count: 2/5', 'Count: 3/5', 'Count: 4/5', 'Count: 5/5']
```

### 2. Asynchronous Call Mode

```python
# Async call - similar output structure to sync call
result = await runner.acall(
    prompt_kwargs={"input_str": "What's the weather in SF and calculate 42 * 3"}
)

print(result.answer)
# Output: San Francisco will be mostly cloudy today with some afternoon sun, reaching about 67 °F (20 °C). 
#         The result of 42 * 3 is 126.
```

### 3. Async Streaming Mode

```python
# Async streaming - real-time event processing
streaming_result = runner.astream(
    prompt_kwargs={"input_str": "Calculate 100 + 50 and count to 3"},
)

# Process streaming events in real-time
async for event in streaming_result.stream_events():
    if isinstance(event, RunItemStreamEvent):
        if isinstance(event.item, ToolCallRunItem):
            print(f"🔧 Calling: {event.item.data.name}")
        elif isinstance(event.item, ToolCallActivityRunItem):
            print(f"📝 Activity: {event.item.data}")
        elif isinstance(event.item, ToolOutputRunItem):
            print(f"✅ Output: {event.item.data.output}")
        elif isinstance(event.item, FinalOutputItem):
            print(f"🎯 Final: {event.item.data.answer}")

# Output:
# 🔧 Calling: calculator
# ✅ Output: The result of 100 + 50 is 150
# 🔧 Calling: counter
# 📝 Activity: Count: 1/3
# 📝 Activity: Count: 2/3
# 📝 Activity: Count: 3/3
# ✅ Output: ['Count: 1/3', 'Count: 2/3', 'Count: 3/3']
# 🎯 Final: The result of 100 + 50 is 150. Counted to 3 successfully.
```

_Set your `OPENAI_API_KEY` environment variable to run these examples._

**Try the full Agent tutorial in Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SylphAI-Inc/AdalFlow/blob/main/notebooks/agents/agent_tutorial.ipynb)

<!-- Please refer to the [full installation guide](https://adalflow.sylph.ai/get_started/installation.html) for more details.
[Package changelog](https://github.com/SylphAI-Inc/AdalFlow/blob/main/adalflow/CHANGELOG.md). -->
View [Quickstart](https://colab.research.google.com/drive/1_YnD4HshzPRARvishoU4IA-qQuX9jHrT?usp=sharing): Learn How `AdalFlow` optimizes LM workflows end-to-end in 15 mins.

Go to [Documentation](https://adalflow.sylph.ai) for tracing, human-in-the-loop, and more.


<!-- * Try the [Building Quickstart](https://colab.research.google.com/drive/1TKw_JHE42Z_AWo8UuRYZCO2iuMgyslTZ?usp=sharing) in Colab to see how AdalFlow can build the task pipeline, including Chatbot, RAG, agent, and structured output.
* Try the [Optimization Quickstart](https://colab.research.google.com/github/SylphAI-Inc/AdalFlow/blob/main/notebooks/qas/adalflow_object_count_auto_optimization.ipynb) to see how AdalFlow can optimize the task pipeline. -->



# Research
[Jan 2025] [Auto-Differentiating Any LLM Workflow: A Farewell to Manual Prompting](https://arxiv.org/abs/2501.16673)
- LLM Applications as auto-differentiation graphs
- Token-efficient and better performance than DsPy


# Collaborations

We work closely with the [**VITA Group** at University of Texas at Austin](https://vita-group.github.io/), under the leadership of [Dr. Atlas Wang](https://www.ece.utexas.edu/people/faculty/atlas-wang), who provides valuable support in driving project initiatives.

 <!-- alongside [Dr. Junyuan Hong](https://jyhong.gitlab.io/),  -->
For collaboration, contact [Li Yin](https://www.linkedin.com/in/li-yin-ai/).

# Hiring

We are looking for a Dev Rel to help us build the community and support our users. If you are interested, please contact [Li Yin](https://www.linkedin.com/in/li-yin-ai/).



<!-- ## Light, Modular, and Model-Agnostic Task Pipeline

LLMs are like water; AdalFlow help you quickly shape them into any applications, from GenAI applications such as chatbots, translation, summarization, code generation, RAG, and autonomous agents to classical NLP tasks like text classification and named entity recognition.

AdalFlow has two fundamental, but powerful, base classes: `Component` for the pipeline and `DataClass` for data interaction with LLMs.
The result is a library with minimal abstraction, providing developers with *maximum customizability*.

You have full control over the prompt template, the model you use, and the output parsing for your task pipeline.


<p align="center">
  <img src="https://raw.githubusercontent.com/SylphAI-Inc/AdalFlow/main/docs/source/_static/images/AdalFlow_task_pipeline.png" alt="AdalFlow Task Pipeline">
</p>

Many providers and models accessible via the same interface:

<p align="center">
  <img src="https://raw.githubusercontent.com/SylphAI-Inc/AdalFlow/main/docs/source/_static/images/multi-providers.png" alt="AdalFlow Model Providers">
</p>

[All available model providers](https://adalflow.sylph.ai/apis/components/components.model_client.html)




Further reading: [How We Started](https://www.linkedin.com/posts/li-yin-ai_both-ai-research-and-engineering-use-pytorch-activity-7189366364694892544-Uk1U?utm_source=share&utm_medium=member_desktop),[Design Philosophy](https://adalflow.sylph.ai/tutorials/lightrag_design_philosophy.html) and [Class hierarchy](https://adalflow.sylph.ai/tutorials/class_hierarchy.html).



## Unified Framework for Auto-Optimization


To optimize your pipeline, simply define a ``Parameter`` and pass it to AdalFlow's ``Generator``.
You use `PROMPT` for prompt tuning via textual gradient descent and `DEMO` for few-shot demonstrations.
We let you **diagnose**, **visualize**, **debug**, and **train** your pipeline.


### **Trainable Task Pipeline**

Just define it as a ``Parameter`` and pass it to AdalFlow's ``Generator``.

<p align="center">
  <img src="https://raw.githubusercontent.com/SylphAI-Inc/AdalFlow/main/docs/source/_static/images/Trainable_task_pipeline.png" alt="AdalFlow Trainable Task Pipeline">
</p>

### **AdalComponent & Trainer**

``AdalComponent`` acts as the 'interpreter'  between task pipeline and the trainer, defining training and validation steps, optimizers, evaluators, loss functions, backward engine for textual gradients or tracing the demonstrations, the teacher generator.

<p align="center">
    <img src="https://raw.githubusercontent.com/SylphAI-Inc/AdalFlow/main/docs/source/_static/images/trainer.png" alt="AdalFlow AdalComponent & Trainer">

</p>
 -->


# Documentation

AdalFlow full documentation available at [adalflow.sylph.ai](https://adalflow.sylph.ai/):
<!-- - [How We Started](https://www.linkedin.com/posts/li-yin-ai_both-ai-research-and-engineering-use-pytorch-activity-7189366364694892544-Uk1U?utm_source=share&utm_medium=member_desktop)
- [Introduction](https://adalflow.sylph.ai/)
- [Full installation guide](https://adalflow.sylph.ai/get_started/installation.html)
- [Design philosophy](https://adalflow.sylph.ai/tutorials/lightrag_design_philosophy.html)
- [Class hierarchy](https://adalflow.sylph.ai/tutorials/class_hierarchy.html)
- [Tutorials](https://adalflow.sylph.ai/tutorials/index.html)
- [Supported Models](https://adalflow.sylph.ai/apis/components/components.model_client.html)
- [Supported Retrievers](https://adalflow.sylph.ai/apis/components/components.retriever.html)
- [API reference](https://adalflow.sylph.ai/apis/index.html) -->


# AdalFlow: A Tribute to Ada Lovelace


AdalFlow is named in honor of [Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace), the pioneering female mathematician who first recognized that machines could go beyond mere calculations. As a team led by a female founder, we aim to inspire more women to pursue careers in AI.

# Community & Contributors

The AdalFlow is a community-driven project, and we welcome everyone to join us in building the future of LLM applications.

Join our [Discord](https://discord.gg/ezzszrRZvT) community to ask questions, share your projects, and get updates on AdalFlow.

To contribute, please read our [Contributor Guide](https://adalflow.sylph.ai/contributor/index.html).

# Contributors

[![contributors](https://contrib.rocks/image?repo=SylphAI-Inc/AdalFlow&max=2000)](https://github.com/SylphAI-Inc/AdalFlow/graphs/contributors)

# Acknowledgements

Many existing works greatly inspired AdalFlow library! Here is a non-exhaustive list:

- 📚 [PyTorch](https://github.com/pytorch/pytorch/) for design philosophy and design pattern of ``Component``, ``Parameter``, ``Sequential``.
- 📚 [Micrograd](https://github.com/karpathy/micrograd): A tiny autograd engine for our auto-differentiative architecture.
- 📚 [Text-Grad](https://github.com/zou-group/textgrad) for the ``Textual Gradient Descent`` text optimizer.
- 📚 [DSPy](https://github.com/stanfordnlp/dspy) for inspiring the ``__{input/output}__fields`` in our ``DataClass`` and the bootstrap few-shot optimizer.
- 📚 [OPRO](https://github.com/google-deepmind/opro) for adding past text instructions along with its accuracy in the text optimizer.
- 📚 [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) for the ``AdalComponent`` and ``Trainer``.

<!-- # Citation

```bibtex

@software{Yin2024AdalFlow,
  author = {Li Yin},
  title = {{AdalFlow: The Library for Large Language Model (LLM) Applications}},
  month = {7},
  year = {2024},
  doi = {10.5281/zenodo.12639531},
  url = {https://github.com/SylphAI-Inc/AdalFlow}
}
``` -->

<!-- # Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SylphAI-Inc/AdalFlow&type=Date)](https://star-history.com/#SylphAI-Inc/AdalFlow&Date) -->
<!--
<a href="https://trendshift.io/repositories/11559" target="_blank"><img src="https://trendshift.io/api/badge/repositories/11559" alt="SylphAI-Inc%2FAdalFlow | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a> -->
