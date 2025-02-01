Development Essentials
======================================
This document will cover how you can set up the AdalFlow codebase and start coding, testing, and documentation.

Set Up
^^^^^^^^^^^^^^^^^^^
We mainly use `poetry <https://python-poetry.org/>`_ for dependency management and virtual environment setup.


To set up ``poetry`` and contribute, please check the following steps:

1. **Clone the Repository:**

   .. code-block:: bash

        git clone https://github.com/SylphAI-Inc/AdalFlow
        cd AdalFlow

2. **Set Up the AdalFlow Dev Environment:**
   The AdalFlow source code, tests, and dependencies are in the ``./adalflow`` directory.
   The ``./adalflow/pyproject.toml`` controls the dependencies for the ``adalflow`` package.
   Use Poetry to install the dependencies and set up the virtual environment:

   .. code-block:: bash

        cd adalflow
        poetry install
        poetry shell

   Test the setup by running the tests at the ``./adalflow`` directory:

   .. code-block:: bash

        pytest tests

3. **Set Up the Root Dev Environment:**
   At the root directory, we have a ``pyproject.toml`` file that controls the dependencies for the root directory.

   .. code-block:: bash

        poetry install
        poetry shell

   This will install all relevant dependencies and the files in /use_cases, /tutorials, and /benchmarks will be using the development version of the ``adalflow`` package.
   You should see output similar to the following:

   .. code-block:: bash

        - Installing adalflow (0.2.5 /Users/liyin/Documents/test/AdalFlow/adalflow)




4. **[Optional] Configure API Keys in the Root Directory:**
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

   This will be helpful for you to run tutorials, use cases, and benchmarks.


Coding
^^^^^^^^^^^^^^^^^^^
Structuring
~~~~~~~~~~~~~~~
It is recommended to check our the structuring in :ref:`part1-structuring` and :doc:`../apis/index`
to understand the codebase structure.

What to code
~~~~~~~~~~~~~~~
Please check the :ref:`part3-contributing-steps` to see some coding examples and steps to contribute to the codebase.

Code Tips
~~~~~~~~~~~~~~~
* Please follow the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_.

* Functions and classes should include standard docstrings and comments. Please refer to `documentation contribution guidelines <./contribute_to_document.html>`_ for standard docstrings.

Copilot
~~~~~~~~~~~~~~~
We suggest you use `GitHub Copilot <https://copilot.github.com/>`_ to help you write code faster and more efficiently.
You can follow this `Guide <https://docs.github.com/en/copilot/using-github-copilot/getting-code-suggestions-in-your-ide-with-github-copilot>`_ to set it up with your IDE.
There are other options like `Cursor <https://www.cursor.com/>`_ and `Tabnine <https://www.tabnine.com/>`_ that you can use as well.

Dependencies
~~~~~~~~~~~~~~~
1. If you want to add any new dependencies to the package, please include them in your PR description to inform us.
2. Since we have already set up the testing automatic workflow in GitHub, please also set your new dependencies in ``./adalflow/pyproject.toml`` file ``[tool.poetry.group.test.dependencies]`` section to avoid dependency errors in our CI/CD workflow.
   In order to correctly add the dependency using ``poetry``, please run

   .. code-block:: bash

      poetry add --group test <package-name>

Testing
^^^^^^^^^^^^^^^^^^^
After you update the code, please make sure your code is well tested before making a pull request.
There is a ``./adalflow/tests`` folder in the project directory to host your unit testing cases.

You might need to install the testing packages using ``poetry``:

For example:

.. code-block:: bash

        poetry install # or
        poetry add --group test


You should name your test files with the following format: ``test_<module_name>.py``.

Activate the virtual environment from `./adalflow` and run the tests:

.. code-block:: bash

    poetry shell
    pytest

To run a specific test file, you can use the following command:

.. code-block:: bash

    pytest tests/test_components.py

For more details on testing, please refer to the `README.md <https://github.com/SylphAI-Inc/AdalFlow/blob/main/adalflow/tests/README.md>`_ under the ``./adalflow/tests`` directory.

Documentation
^^^^^^^^^^^^^^^^^^^
Please refer to the `README.md <https://github.com/SylphAI-Inc/AdalFlow/blob/main/docs/README.md>`_ under the ``./docs`` directory for more details on how to contribute to the documentation.
