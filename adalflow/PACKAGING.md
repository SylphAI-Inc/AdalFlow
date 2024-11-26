# Poetry Packaging Guide
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
pip install "dist/adalflow-0.1.0b1-py3-none-any.whl[openai,groq,faiss]"
```


## Update the version

1. Update the version in `pyproject.toml`
2. Add the version number in `adalflow/__init__.py`
3. Build the package
4. Test the package locally
5. Push the changes to the repository
6. Ensure to run `poetry lock --no-update` in the root directory (project-level) to update the lock file for other directories such as `tutorials`, `use_cases`, `benchmarks`, etc.
7. Update the `CHANGELOG.md` file with the new version number and the changes made in the new version.

## TODO: we need to automate the version update process. Help is appreciated.
