Contributing process
=======================================
Welcome to the AdalFlow community! We tried our best to make the process as simple and as clear as possible. We are open to any suggestions and advice to improve the process.
Please feel free to contact us on `Discord <https://discord.com/invite/ezzszrRZvT>`_.


Quick Start
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. The `Github issues < https>`_ is the best place to pick up a first task. Simply look for a task labeled `good first issue`.
2. The follow the `Code Contribution Guidelines <./contribute_to_code.html>`_ to start setting up your environment, coding and testing.
3. Last, you can follow the `Documentation Contribution Guidelines <./contribute_to_document.html>`_ to write documentation for your code.
4. Check out the last section on `PR & Review Process <#pr-review-process>`_ to complete the review and iteration process. We are trying out best to maximize both your learning and the quality of the library.


.. note::

   You can use 👍 to indicate that you want a particular issue to be addressed.


1. Structuring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To go deeper, we will explain our repo structure, issue and label system.

..  what to contribute(with examples), contributing steps with proposal/discussion/coding/testing/documentation/pr/review process.
.. The coding and testing will be discussed more in details in `Code Contribution Guidelines <./contribute_to_code.html>`_ and the documentation will be discussed in `Documentation Contribution Guidelines <./contribute_to_document.html>`_.

Repo Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We have a clear structure for the repo. The repo is divided into 6 subdirectories:

.. code-block:: text

   .
   ├── .github/
   ├── adalflow/
   │   ├── adalflow/
   │   ├── tests/
   |   ├── pyproject.toml
   ├── docs/
   |   |── pyproject.toml
   ├── tutorials/
   ├── use_cases/
   ├── benchmarks/
   ├── notebooks/
   |   ├── tutorials/
   |   ├── use_cases/
   |   ├── benchmarks/
   ├── .env_example
   ├── .gitattributes
   ├── .gitignore
   ├── .pre-commit-config.yaml
   ├── CNAME
   ├── LICENSE.md
   ├── README.md
   ├── SETUP.md
   ├── poetry.lock
   ├── pyproject.toml

The `/adalflow` directory contains the source code for the `AdalFlow` library, it has its soource code and tests, along with its own `pyproject.toml` file.
The `docs` directory contains the documentation for the `AdalFlow` library, it has its own `pyproject.toml` file too.
We use `reStructuredText` for the documentation. Please refer to `Documentation Contribution Guidelines <./contribute_to_document.html>`_ for more details.

Besides, we have `tutorials`, `use_cases`, and `benchmarks` directories for the tutorials, use cases, and benchmarks of the `AdalFlow` library.
`notebooks` directory contains all notebooks that are used across `tutorials`, `use_cases`, and `benchmarks`.


Issue & Label System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We use issues to manage our bugs, features, and discussions.
We carefully designed 13 labels to categorize these issues.


.. figure:: ../_static/images/adalflow_issues.png
   :alt: AdalFlow Issues
   :align: center
   :width: 700px

   **Type**: The type of the issue, such as bug, feature, or discussion.

We use three categories of labels in parallel:

* Type of issue: There are 7 types of issues. We use `[adalflow]` to indicate the issue is related to `AdalFlow` source code under the `/adalflow` directory. Under this directory, we have two subdirectories: `adalflow` for the source code and `tests` for the test code.  You can suggest `integration`, `improvement`, `core feature`, `bug` here. Additionally, you can `documentation` for things located in the `/docs`, `/tutorials`, and `/notebooks` directories. "new use cases/benchmarks" is for new use cases or benchmarks located in the `/use_cases` and `/benchmarks` directories. "question" is for general questions.
* How to proceed: There are 3 types of issues. We use `good first issue` to indicate the issue is suitable for new contributors. We use `wontfix` to indicate the issue is not suitable for the library. We use `duplicate` to indicate the issue is a duplicate of another issue.
* Priority: There are 3 types of issues. We use `P0` to indicate the issue is the highest priority. We use `P1` to indicate the issue is the second highest priority. We use `P2` to indicate the issue is the lowest priority.


.. list-table:: Type of issue, How to proceed, and Priority
   :header-rows: 1
   :widths: 40 70 10

   * - Type of issue (7 labels)
     - How to proceed (3 labels)
     - Priority
   * - [adalflow] suggest integration
     -
     -
   * - [adalflow] suggest improvement
     - wontfix
     - P0
   * - [adalflow] suggest core feature
     - good first issue
     -
   * - new use cases/benchmarks
     - duplicate (aggregate) and close one
     - P1
   * - [adalflow] bug
     -
     - P2
   * - question
     -
     -
   * - documentation
     -
     -

How to assign priority?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Besides our team to mark the priority with our best judgement, we allow the community to give us more signals on the priority.
You can use 👍 to indicate the importance of a particular issue to you.
We will take the `# of 👍 / time_period` as a signal to the priority too.


2. What to contribute
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This section explains more details on how each issue will relate to our codebase. We will list some example prs to help you understand better.
The following table will provide an quick overview. We will provide more details in each subsection on each type of contributions.


.. list-table:: What to Contribute (by 7 Labels) and Example PRs
   :header-rows: 1
   :widths: 20 50 30

   * - Label
     - Contribution Suggestions
     - Example PRs
   * - [adalflow] bug
     - Fix bugs reported in issues or identified during testing.
     - Fix data processing errors or incorrect retriever outputs.
   * - [adalflow] suggest integration
     - Add new integrations with external tools or data sources.
     - Integrate a cloud data store as a new retriever and include usage documentation.
   * - [adalflow] suggest improvement
     - Enhance existing features for better performance or usability.
     - Optimize `text_splitter` or improve `LocalDB` handling efficiency.
   * - [adalflow] suggest core feature
     - Develop new core functionalities for the library.
     - Add support for advanced retriever capabilities or new storage backends.
   * - new use cases/benchmarks
     - Design benchmarks or propose new use cases for `adalflow`.
     - Add retriever performance benchmarks or examples of practical applications.
   * - documentation
     - Improve documentation and tutorials for better user understanding.
     - Create Colab notebooks, add integration examples, or refine tutorials.
   * - question
     - Answer user queries or provide clarifications about the library.
     - Add a Q&A section to the documentation or examples for common user issues.


3. Contributing Steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

4. PR & Review Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You are always welcomed to contribute even if you've never participated in open source project before.
Here is the basic contribution process:

Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When contributing, please note:
LightRAG separates the source code environment and documentation environment.

* To activate the code environment, you should run ``poetry install`` and ``poetry shell`` under ``./lightrag``. The ``./lightrag/pyproject.toml`` contains the dependencies for the ``LightRAG`` package.

* To activate the documentation environment, you can run ``poetry install`` and ``poetry shell`` under ``.``. The ``./pyproject.toml`` controls documentation dependencies.

Find a direction to work on
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The team builds ``LightRAG`` based on latest researches and product cases. But you might have your own task to apply ``LightRAG``.
Therefore, you can extend ``LightRAG`` and add any new features you believe will solve yours or others' problems.
If you don't have any idea yet, you can:

* Check the `existing issues <https://github.com/SylphAI-Inc/LightRAG/issues>`_ and see if there is anyone you know how to fix or you'd love to fix.

* Join us on `Discord <https://discord.com/invite/ezzszrRZvT>`_. We are glad to discuss with you and know what you are interested in here.

Figure out the scope of your change
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Small:** Most of the pull requests are small. If your change is small, such as fixing a line of bug, please go ahead to push it.

**Big:** But if you are making a new feature, or planning to push a large change, it is recommended to contact us on `Discord <https://discord.com/invite/ezzszrRZvT>`_ first.

**Unknown:** If you have no idea how big it will be, we are here to help you. Please post your idea on `issues <https://github.com/SylphAI-Inc/LightRAG/issues>`_. We will read it carefully and get back to you.

Add your code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Please check our `code contribution guidelines <./contribute_to_code.html>`_ to work with code.

Pull requests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**WIP PR:** If you are working on an in pull request that is not ready for review, you can create a PR with **"[WIP]"** to inform us that this PR is a draft **“work in progress”**.

**Finished PR:** You can name your finished PR as **"[New Retriever Integration]"** for example.
We will carry out code review regularly and provide feedbacks as soon as possible.
Please iterate your PR with the feedbacks. We will try our best to reduce the revision workload on your side.
Once your PR is approved, we will merge the PR for you.
If you have any concerns about our feedbacks, please feel free to contact us on `Discord <https://discord.com/invite/ezzszrRZvT>`_.

Writing Documentation
----------------------------
It is a good practice to submit your code with documentations to help the ``LightRAG`` team and other developers better understand your updates.
Please see our `documentation contribution guidelines <./contribute_to_document.html>`_ for more details on ``LightRAG`` documentation standard.




.. .. admonition:: Resources
..    :class: highlight
