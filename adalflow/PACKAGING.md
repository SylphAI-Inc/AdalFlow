#Poetry Packaging Guide
## Development

To install optional dependencies, use the following command:

```bash
poetry install --extras "openai groq faiss"
```
Install more extra after the first installation, you will use the same command:

```bash
poetry install --extras "anthropic cohere google-generativeai pgvector"
```

## Extra Dependencies
Add the optional package in dependencides.

Build it locally:
```bash
poetry build
```

Test the package locally:

Better to use a colab to update the whl file and test the installation.

```bash
pip install "dist/lightrag-0.1.0b1-py3-none-any.whl[openai,groq,faiss]"
```
