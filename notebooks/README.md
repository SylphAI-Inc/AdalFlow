This is where all of our colab notebooks will be tracked as `ipynb` files.

There are still other notebooks in both `tutorials/` and `use_cases` directories that will be migrated here.

## Objective

Jupyter notebooks/colabs will complement our documents at [documentation website](https://adalflow.sylph.ai) and their source code at either `tutorials/` or `use_cases/`. These are designed to be less verbose than our documents and showcase the code and results.


## Structure

We provided a colab template at `notebooks/adalflow_colab_template`. You can make a copy using:

`cp notebooks/adalflow_colab_template.ipynb notebooks/your_new_colab.ipynb`.

The template consists of three parts:

1. Welcome to AdalFlow with library intro, outline, and installation along with environment setup.
2. Content section of your notebook. Link to Next that users can look at.
3. Issues and Feedback.

#  Tests

## Use kernel first if you are developing something new

If you want to use an ikernel in .ipynb to test notebooks

You can use the following command to install the kernel at the root of the project:

```poetry run python -m ipykernel install --user --name my-project-kernel```

## If a new package needs to be released and tested on the changes

You can go to adalflow dir and run

```bash
poetry build
```

And use

```bash
pip install your_path/dist/adalflow-0.1.0-py3-none-any.whl
```

to install the package.

## Link

Your colab link will be:

`https://colab.research.google.com/github/SylphAI-Inc/AdalFlow/blob/main/notebooks/your_new_colab.ipynb`
