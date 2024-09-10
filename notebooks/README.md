This is where all our colab notebookes are gonna be tracked as `ipynb` files.

There are still other notebooks in both `tutorials/` and `use_cases` directories that we will migrate to here.

## Objective

Jupyter notebooks/colabs will be in complementary to documents on our [documentation website](https://adalflow.sylph.ai) and its source code at either `tutorials/` or `use_cases/`. It is designed to have less text compared with documents and more showcasing the code and results.


## Structure

We provided a colab template at `notebooks/adalflow_colab_template` that you can make a copy use:

`cp notebooks/adalflow_colab_template.ipynb notebooks/your_new_colab.ipynb`.

The template consists of three parts:

1. Welcome to AdalFlow with library intro, outline, and installation along with environment setup.
2. Content section of your notebook. Link to Next that users can look at.
3. Issues and Feedback.


## If you need to use dev api

You can go to adalflow dir and do

```bash
poetry build
```

And use

```bash
pip install your_path/dist/adalflow-0.1.0-py3-none-any.whl
```

to install the package.
