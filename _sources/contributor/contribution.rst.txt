LightRAG Contribution Guide
=======================================
``LightRAG``'s contribution process is similar to most open source projects on GitHub. We encourage new project ideas and the communication between ``LightRAG`` team, developers and the broader community.
Please don't forget to join us on `Discord <https://discord.com/invite/ezzszrRZvT>`_.

Contribution Process
----------------------------
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




.. admonition:: Resources
   :class: highlight

   