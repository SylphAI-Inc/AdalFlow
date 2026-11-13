# AdalFlow Codebase Summary

AdalFlow is a PyTorch-style Python library for building and auto-optimizing LLM workflows including chatbots, RAG pipelines, and agents. Named after Ada Lovelace, it powers the [AdaL CLI](https://sylph.ai) and provides model-agnostic building blocks with built-in auto-prompt optimization via textual gradient descent and few-shot bootstrap.

## Tech Stack

- **Language**: Python 3.9 - 3.14
- **Package Management**: Poetry
- **Core Dependencies**: pydantic, jinja2, tiktoken, numpy, PyYAML, nest-asyncio, diskcache, backoff
- **LLM Providers** (optional): OpenAI, Anthropic, Google Generative AI, Groq, Cohere, Ollama, Together, Mistral, Fireworks, AWS Bedrock, Azure AI
- **Vector Databases**: FAISS, pgvector, LanceDB, Qdrant
- **Testing**: pytest, pytest-asyncio, pytest-mock
- **Docs**: Sphinx with pydata theme, deployed to GitHub Pages
- **Linting/Formatting**: Black, Ruff, pre-commit hooks

## Repository Structure

```
/
├── adalflow/                  # Core library package
│   ├── adalflow/              # Main Python package
│   │   ├── core/              # Base abstractions (Component, Generator, DataClass, ModelClient, Retriever)
│   │   ├── components/        # Implementations
│   │   │   ├── model_client/  # OpenAI, Anthropic, Groq, Ollama, Bedrock, Azure, etc.
│   │   │   ├── retriever/     # FAISS, LanceDB, Qdrant, Postgres, BM25, LLM retriever
│   │   │   ├── output_parsers/# JSON, YAML, DataClass parsers
│   │   │   ├── agent/         # Agent, Runner, ReActAgent
│   │   │   ├── data_process/  # TextSplitter
│   │   │   └── memory/        # Conversation memory
│   │   ├── optim/             # Optimization framework
│   │   │   ├── text_grad/     # Textual gradient descent optimizer
│   │   │   ├── few_shot/      # Bootstrap few-shot optimizer
│   │   │   └── trainer/       # Training loop
│   │   ├── eval/              # Evaluators (LLM-as-judge, answer match, recall)
│   │   ├── datasets/          # Data loaders (GSM8K, BigBench Hard, HotpotQA, TREC)
│   │   ├── tracing/           # Observability (spans, traces, MLflow integration)
│   │   ├── database/          # SQLAlchemy + pgvector for document storage
│   │   ├── utils/             # Helpers (setup_env, config, serialization, registry)
│   │   └── apps/              # Permission handlers (CLI, FastAPI for human-in-the-loop)
│   └── tests/                 # Unit tests (50+ test files)
├── docs/                      # Sphinx documentation source
├── notebooks/                 # Jupyter notebooks (agents, evaluation, integrations)
├── tutorials/                 # Tutorial scripts
├── use_cases/                 # Example applications
│   ├── configs/               # YAML configs (rag.yaml, etc.)
│   ├── rag/                   # RAG examples
│   ├── classification/        # Classification tasks
│   └── question_answering/    # GSM8K, BigBench Hard benchmarks
├── benchmarks/                # Performance benchmarks
├── .github/                   # CI workflows, issue/PR templates
├── pyproject.toml             # Root project config (Poetry, dev deps)
├── Makefile                   # Dev commands (setup, format, lint, test)
└── .pre-commit-config.yaml    # Pre-commit hooks
```

## Architecture

### Core Abstractions (PyTorch-like)

| Abstraction | Description |
|---|---|
| **Component** | Base class for all pipeline elements; supports `train()`, `trace()`, `use_teacher()`, nested subcomponents |
| **GradComponent** | Extends Component with `forward()` and `backward()` for auto-differentiation |
| **Generator** | Orchestrates Prompt -> ModelClient -> Output parsing for LLM calls |
| **DataClass** | Base for LLM input/output schemas with `__input_fields__` and `__output_fields__` |
| **Parameter** | Trainable prompt and demo parameters (`ParameterType.PROMPT`, `ParameterType.DEMOS`) |
| **ModelClient** | Abstract interface for LLM providers |
| **Retriever** | Abstract interface for vector/text search backends |

### Call Interface

- **`call()`** / **`acall()`** -- Sync and async inference
- **`forward()`** -- Training / gradient computation path

### Optimization Pipeline

- **AdalComponent** connects a task pipeline to the Trainer by defining `prepare_task`, `prepare_eval`, and `prepare_loss`
- **Trainer** runs the optimization loop supporting textual gradient descent and few-shot bootstrap
- **TGDOptimizer** performs textual gradient descent on prompt parameters
- **BootstrapFewShot** optimizes few-shot demonstration selection

### Agent System

- **ReActAgent** implements the Reasoning + Acting pattern with tool use
- **Runner** orchestrates agent execution (sync, async, streaming)
- **ToolManager** manages FunctionTool definitions
- **PermissionManager** enables human-in-the-loop approval for tool calls (CLI or FastAPI)

### Data and Storage

- **LocalDB** provides in-memory document storage with transform pipelines
- **Document** model with id, text, metadata, and embedding fields
- **SQLAlchemy integration** with pgvector for persistent vector search

### Tracing and Observability

- OpenAI-compatible span/trace format
- MLflow integration for experiment tracking
- Configurable via `ADALFLOW_TRACING_API_KEY` and `ADALFLOW_DISABLE_TRACING` env vars

## Component Flow

```
User Application (use_cases, tutorials, notebooks)
         │
         ▼
Runner / Agent / Component
         │
    ┌────┼────────────────┐
    ▼    ▼                ▼
Generator  ToolManager  PermissionManager
    │         │           (human-in-the-loop)
    ▼         ▼
ModelClient  Retriever
(OpenAI,     (FAISS, BM25,
 Anthropic,   LanceDB,
 Ollama...)   Qdrant...)
    │
    ▼
Tracing (spans, MLflow)

Training Path:
Trainer ─► AdalComponent ─► task Component + eval_fn + loss_fn
```

## Build and CI

| Makefile Target | Description |
|---|---|
| `make setup` | Install dependencies and pre-commit hooks |
| `make format` | Run Black and Ruff formatting |
| `make lint` | Run Ruff linting checks |
| `make test` | Run pytest suite |
| `make clean` | Remove caches and build artifacts |

### GitHub Actions Workflows

- **python-test.yml** -- Runs pytest on PRs to `main` across Python 3.9, 3.10, 3.11, 3.12
- **documentation.yml** -- Builds Sphinx docs on pushes to `release` and deploys to GitHub Pages
- **release.yml** -- Publishes to PyPI on version tags (`v*.*.*`)

## Public API Highlights

The library is imported as:

```python
import adalflow as adal
```

Key exports include `Component`, `Generator`, `Embedder`, `Retriever`, `Agent`, `Runner`, `Parameter`, `AdalComponent`, `Trainer`, `TGDOptimizer`, `DataClassParser`, `JsonOutputParser`, `YamlOutputParser`, `OpenAIClient`, `GroqAPIClient`, `AnthropicAPIClient`, `OllamaClient`, `setup_env`, and `get_logger`.

## Configuration

- **Environment**: `setup_env(dotenv_path=".env")` loads API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.)
- **Component config**: `new_component(config)` builds components from nested dicts with `component_name` and `component_config`
- **YAML configs**: Use case configurations in `use_cases/configs/` (e.g., `rag.yaml`)
