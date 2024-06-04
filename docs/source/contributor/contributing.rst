Contributing
===============================================

.. contents::
   :local:
   :depth: 2

.. _Writing documentation:

Writing Documentation
---------------------------

- **User-Facing Documentation**: Found on the main docs site. These include tutorials, guides, and usage documentation meant for end users.
- **Developer Documentation**: Located within the repository's READMEs and the ``docs/`` directory. These documents are more technical and intended for contributors and maintainers.

This section is about user-facing documentation.

LightRAG uses `Sphinx <https://www.sphinx-doc.org/en/master/>`_ for documentation, leveraging both `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ and Sphinx's `autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_ feature to pull docstrings from code and organize them through ``.rst`` files. Our documentation is split into:

Souce Code Docstring Standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sphinx automatically pulls docstrings from source code and uses them as the docs in API reference. For clarity and consistency, we have a standard for all the code contributors.

Aligning with Pytorch, LightRAG uses the `Google style with Sphinx <https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html>`_ for formatting docstrings `(detailed styles) <https://google.github.io/styleguide/pyguide.html>`_, emphasizing **docstring** and **type control** to guarantee the document and code quality.


Setup & Build Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Clone the GitHub Project**

.. code-block:: bash

    git clone https://github.com/SylphAI-Inc/LightRAG.git

**2. Install Necessary Packages**

LightRAG's documentation style is `pydata_sphinx_theme <https://pydata-sphinx-theme.readthedocs.io/en/stable/>`_ (version: 0.15.2).

Install by ``pip``:

.. code-block:: bash

    cd docs
    pip install -r requirements.txt 

Install by ``poetry`` along with all other dependencies for LightRAG:

.. code-block:: bash

    poetry install

**3. Build the Documentation**

.. code-block:: bash

    cd docs
    make html


**conf.py**

This file (``docs/source/conf.py``) contains configurations used by Sphinx, including extensions, templates, HTML theme, and language settings.

**Source Code Doc-string** 

Follow `Google style docstrings <https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html>`_ to update your source code docstrings. Limit lines to **80** characters for better readability in various environments. 

**RST Files**: Directly edit ``.rst`` files for broader changes or new sections. Use the ``.. toctree::`` directive to link documents.

The ``.rst`` files are in the ``docs/source``. The majority of ``.rst`` files in the ``docs/source/apis`` are generated automatically from the Python code docstrings using ``sphinx-apidoc``.

To shorten the doc generating process, please remove the files that is not included in your project.

The Sphinx build will show warnings but the docs will still be completed.

If you have a module folder containing code, for example, ``components/``, please add the following line to the ``docs/Makefile`` in the ``apidoc:`` section.

.. code-block:: bash
    
    @sphinx-apidoc -o $(APIDOCOUTDIR)/components ../components --separate --force


**4. View the Documentation Locally**

After building, open ``docs/build/html/index.html`` in a web browser. If you face issues with local resources, such as the browser prohibits loading the web page correctly, run a local server:

.. code-block:: bash

    cd docs/build/html
    python -m http.server 8000 <path_to_html_output>

Then navigate to the corresbonding site in your browser.



Adding Documentation Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure the documentation remains up-to-date, LightRAG uses Sphinx's Doctest extension. Add ``.. testcode::`` to your ``.rst`` files or docstrings and run ``make doctest`` to test your documentation snippets.

To manually run these tests, run:

.. code-block:: bash

    cd docs
    make doctest



Commit Changes
~~~~~~~~~~~~~~~~~~~~~~~~~

After making changes, commit the ``.rst`` and source files, avoiding the ``docs/build`` directory, and push them to your GitHub fork for review.

