# AGENTS.md

Notes for agents (and humans) working with AdalFlow's **local (transformers) model
clients**, measured on a real NVIDIA Tesla T4 (16GB, Turing, `sm_75`).

## Local (transformers) model client install

AdalFlow is primarily a hosted-API framework (OpenAI/Anthropic/Cohere/Groq/...).
If you want to use the **local** Hugging Face `transformers` path in
`adalflow/adalflow/components/model_client/transformers_client.py`
(`TransformerEmbedder`, `TransformerReranker`, `TransformerLLM`), note the following,
which is not documented elsewhere:

- There is no `transformers` extra in `adalflow/pyproject.toml`
  (`[tool.poetry.extras]` only lists a `torch` extra). You need to install
  `torch`, `transformers`, and `accelerate` yourself, matching your CUDA driver
  (e.g. a machine with driver-level CUDA 12.4 needs a `cu124` torch build —
  the default PyPI index can resolve to a newer CUDA build that reports
  `torch.cuda.is_available() == False` on such a machine).
- `openai` is marked `optional = true` in `adalflow/pyproject.toml`, but it is
  required at import time even for the local-only path: `adalflow/adalflow/core/__init__.py`
  imports `generator.py`, which does a top-level
  `from openai.types.responses import ResponseCompletedEvent`. Without
  `pip install openai`, `from adalflow.core.embedder import Embedder` fails with
  `ModuleNotFoundError: No module named 'openai'`, even if you only intend to use
  the local `transformers` embedder/reranker.
- After installing, verify GPU visibility with
  `python -c "import torch; print(torch.cuda.is_available())"` before running
  local-model examples.

## Local GPU inference surface (measured on Tesla T4)

- **`TransformerEmbedder` (`thenlper/gte-base`) runs CPU-only, even when CUDA is
  available.** Unlike its sibling `TransformerReranker`, it never calls
  `.to(device)` on the model or the tokenizer output
  (`transformers_client.py`, `TransformerEmbedder.__init__`/`infer_gte_base_embedding`).
  Measured: `model_param_device=cpu`, peak VRAM `0.0GB` on a T4 box with CUDA
  available. If you need GPU-accelerated local embeddings, use a different
  client (e.g. `sentence-transformers` directly) or move the model/inputs to
  `cuda` yourself.
- **`TransformerReranker` (`BAAI/bge-reranker-base`) does run on GPU** via
  `get_device()` + `self.model.to(device)` (`TransformerReranker.init_model`),
  but it is loaded in **fp32** with no dtype knob exposed. On a Tesla T4
  (Turing, `sm_75`), fp16 measured **~4.76x faster** than the fp32 default
  (0.607s -> 0.128s warm, batch of 64 documents) and used **about half the
  VRAM** (1.57GB -> 0.80GB). There is currently no supported way to request
  fp16 for this client.
- **`TransformerLLM` hardcodes `torch_dtype=torch.bfloat16`** in both its
  pipeline path and its `AutoModelForCausalLM` path. Turing GPUs (T4, `sm_75`)
  have no bf16 tensor cores, so this dtype is not accelerated on T4-class
  hardware; fp16 is the correct choice there instead.
- Overall, on a Tesla T4 the only local surface that actually uses the GPU is
  the reranker; the local embedder silently stays on CPU. Peak VRAM across the
  measured local surfaces stayed under ~1.6GB, well within a 16GB T4.

These notes describe current, unchanged behavior of the code as of commit
`810de99`; no default values or code paths were modified to produce them.
