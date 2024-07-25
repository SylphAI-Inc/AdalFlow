Contribute to Code
======================================
This document will cover how you can contribute to lightRAG codebase.

Set Up
^^^^^^^^^^^^^^^^^^^
The current ``LightRAG`` code contribution supports `poetry <https://python-poetry.org/>`_ setup only. The team is working on optimizing the library and will get back to support more environment soon.
If you are only interested in using ``LightRAG`` as a package, please check our `installation guide <https://lightrag.sylph.ai/get_started/installation.html#install-lightrag>`_.

To set up ``poetry`` and contribute, please check the following steps:

1. **Clone the Repository:**

   .. code-block:: bash

        git clone https://github.com/SylphAI-Inc/LightRAG
        cd LightRAG

2. **Configure API Keys:**

   Copy the example environment file and add your API keys:

   .. code-block:: bash

        cp .env.example .env
        # example API keys:
        # OPENAI_API_KEY=YOUR_API_KEY_IF_YOU_USE_OPENAI
        # GROQ_API_KEY=YOUR_API_KEY_IF_YOU_USE_GROQ
        # ANTHROPIC_API_KEY=YOUR_API_KEY_IF_YOU_USE_ANTHROPIC
        # GOOGLE_API_KEY=YOUR_API_KEY_IF_YOU_USE_GOOGLE
        # COHERE_API_KEY=YOUR_API_KEY_IF_YOU_USE_COHERE
        # HF_TOKEN=YOUR_API_KEY_IF_YOU_USE_HF

3. **Install Dependencies:**

    The ``./lightrag/pyproject.toml`` controls the dependencies for the ``LightRAG`` package.
    Use Poetry to install the dependencies and set up the virtual environment:

   .. code-block:: bash
        cd lightrag
        poetry install
        poetry shell

Codebase Structure
^^^^^^^^^^^^^^^^^^^
It is recommended to check our `LightRAG codebase structure <https://lightrag.sylph.ai/developer_notes/index.html>`_ and current `API references <https://lightrag.sylph.ai/apis/index.html>`_ to familiarize yourself with the directories and paths before contributing.

Code Examples
^^^^^^^^^^^^^^^^^^^
We want to support you with our best. We have included code samples in the `tutorial <https://lightrag.sylph.ai/developer_notes/index.html>`_ for you to refer to.

We inlcude a list of potential samples(`We are working in progress to add more`):

- `ModelClient integration <https://lightrag.sylph.ai/developer_notes/model_client.html#model-inference-sdks>`_. This document will help if you want to add new models not included in our codebase.
- `Retriever Integration <https://lightrag.sylph.ai/developer_notes/retriever.html#retriever-in-action>`_. We provide different retrivers but you can create more.

Code Tips
^^^^^^^^^^^^^^^^^^^
* When writing code, it is appreciated to include any important docstrings and comments. Please refer to `documentation contribution guidelines <./contribute_to_document.html>`_ for standard docstrings.
* LightRAG is a Python library and if you could follow the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_, the codebase will be more consistent.

Dependencies
^^^^^^^^^^^^^^^^^^^
If you want to add any new dependencies to the package, please include them in your PR description to inform us.
Since we have already set up the testing automatic workflow in GitHub, please also set your new dependencies in
``./lightrag/pyproject.toml`` file ``[tool.poetry.group.test.dependencies]`` section to avoid dependency errors in our CI/CD workflow.

In order to correctly add the dependency using ``poetry``, please run

.. code-block:: bash

        poetry add --group test <package-name>

Testing
^^^^^^^^^^^^^^^^^^^
After you update the code, please make sure your code is well tested before making a pull request.
There is a ``./lightrag/tests`` folder in the project directory to host your unit testing cases.

You might need to install the testing packages using ``poetry``:

For example:

.. code-block:: bash

        poetry add --group test unittest
        poetry add --group test pytest
        poetry add --group test mypy


All the test scripts should start with ``test_``. For example, run the individual test for ``components`` with:

.. code-block:: bash

    python lightrag/tests/test_components.py
