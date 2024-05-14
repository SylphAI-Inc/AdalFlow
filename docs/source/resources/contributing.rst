Contributing
=================================

Thank you very much for your interest in contributing to LightRAG. If you want to contribute to the LightRAG documentation, please refer to the following steps.

Contributing to Documentation
-----------------------------

For more detailed instructions on contributing to the documentation, please check `the repo <https://github.com/SylphAI-Inc/LightRAG/blob/xiaoyi/docs/README.md>`_.

1. Install Necessary Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install Sphinx and the required Sphinx theme:

.. code-block:: bash

    pip install sphinx
    pip install sphinx-rtd-theme

2. Build the Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Navigate to the ``docs`` directory within your project folder and compile the documentation:

.. code-block:: bash

    cd docs
    make html

This command generates the documentation from the source files and outputs HTML files to ``docs/build/html``.

3. View the Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^

After building the documentation, you can view it by opening the ``index.html`` file located in ``docs/build/html``. You can open this file in any web browser to review the generated documentation.

4. Editing Tips for Sphinx Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To effectively edit the LightRAG documentation, you have several options depending on your specific needs:

**Directly Edit an Existing .rst File**

Locate the ``.rst`` file you want to edit within the ``docs/source`` directory. You can modify the content directly in any text editor. We are using `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ as the language. For formatting help, refer to the reStructuredText Quickstart Guide:

- `Quickstart <https://docutils.sourceforge.io/docs/user/rst/quickstart.html>`
- `reStructuredText Markup Specification <https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html>`

**Create a New .rst File**

If you need to add a new section or topic:

- Create a new ``.rst`` file in the appropriate subdirectory within ``docs/source``.
- Write your content following `reStructuredText syntax <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`.
- If you are creating a new section, ensure to include your new file in the relevant ``toctree`` located usually in ``index.rst`` or within the closest parent ``.rst`` file, to make it appear in the compiled documentation.

**Convert a Markdown File to .rst Using Pandoc**

To integrate content written in Markdown into the Sphinx project, use Pandoc to convert it to ``.rst`` format:

Pandoc is a package to transform the files to ``.rst`` files.

- First, install Pandoc with Homebrew:

  .. code-block:: bash

      brew install pandoc

- You might also want to combine the `sphinx extensions <https://www.sphinx-doc.org/en/master/usage/extensions/index.html>`_ in your ``doc/source/conf.py`` for a better layout.
- Then run:

  .. code-block:: bash

      pandoc -s <input .md file> -o <path/to/target_rst_file>

  For example, in the root directory ``pandoc -s README.md -o docs/source/get_started/introduction.rst``. This command will take content from ``README.md`` and create an ``introduction.rst`` file in the specified directory.

**Note:** Most documentation should be written in the code as comments, which will be converted to docs automatically.

5. After editing
^^^^^^^^^^^^^^^^

Once you've made your edits, rebuild the documentation to see your changes:

- Clean previous builds:

  .. code-block:: bash

      make clean

- Generate new HTML documentation:

  .. code-block:: bash

      make html

  We have already included the necessary extensions in the configuration (conf.py), therefore, when you update the code, simply do the rebuilding by ``make html``, the documentation will be updated.

- You can preview the documentation locally by opening ``docs/build/html/index.html``
- Ensure to commit your changes, push them to the GitHub repository and submit a pull request to make them available to others.
