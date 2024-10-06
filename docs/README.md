# AdalFlow Documentation Guide

## Content Overview

- [How the Documentation Works](#how-the-documentation-works)
- [Setup](#setup)
- [File Structure](#file-structure)
- [How to Edit the Documentation](#how-to-edit-the-documentation)

## How the Documentation Works

AdalFlow uses [Sphinx](https://www.sphinx-doc.org/en/master/) and [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) for our documentation. Sphinx primarily reads configurations from a Python script (`conf.py`), pulls documentation from comments in the code (via the `autodoc` extension), and organizes content through its table of contents hierarchy defined in `.rst` files.

## Setup

### **1. Clone the Github Project**

`git clone https://github.com/SylphAI-Inc/AdalFlow.git`

### **2. Install Necessary Packages**

AdalFlow documentation uses `poetry` and `pyproject.toml` to manage the dependencies. You can install all the necessary packages by running:

```

poetry lock
poetry install
```

**NOTE:** The default versions of `sphinx-build` and `poetry` that come with Linux may be out of date for AdalFlow's documentation. Using "pip install" will get the latest versions of those tools.

<!--
All the packages are manged in the project's ``pyproject.toml`` file in the doc dependencies section. You can install all the necessary packages by running:

```
poetry install --with doc
```

**Use `pip`:**

Or you can use `pip` to install the necessary packages listed in ``requirements.txt``:

```
pip install -r requirements.txt
``` -->

### **3. Build the Documentation**
Go to the `docs` directory and run the following command to build the documentation:

```python
cd docs
make html
```

Build with more verbose output:

```
sphinx-build -b html source build -v
```

### **4. View the Documentation**

After building the documentation, you can use any browser to view it by opening the `index.html` file located in `docs/build/html`.

Some browsers restrict loading local resources like CSS for security reasons. In this case, try to use a local server to serve the file in the `build/html` directory. Run the following code:

```
cd docs/build/html
python -m http.server  # Run this in your build/html directory
```

You will find the port shown in the terminal, e.g. `Serving HTTP on :: port 8000 (http://[::]:8000/) ...`. In this case, you should open `http://127.0.0.1:8000/` in your browser and you will see the documentation.

## File Structure

### **conf.py**

The `docs/source/conf.py` controls the configurations used by Sphinx to build the documentation, including the project-related information, [sphinx extensions](https://www.sphinx-doc.org/en/master/usage/extensions/index.html), templates configuration, html theme, patterns to exclude, language configuration, project path setup, etc.

### **index.rst**

The `docs/source/index.rst` is the root document for Sphinx-generated documentation("homepage" for the documentation site). It includes the `toctree` that defines the documentation hierarchical structure(sections/chapters). It also links to other `.rst` files that users can navigate through.

For example, in the `index.rst`, the `:caption: Get Started` corresponds to the section name of the documentation site. `installation` and `introduction` are the detailed pages.

```python
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Get Started

   get_started/installation
   get_started/introduction
```

<!-- ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/bf4570a3-1b74-45d8-8b3b-10221ec99a40/edd3bba3-265a-44cc-94d5-5ab222f9cb71/Untitled.png) -->

### **Existing Sections**

Existing sections include:

`get_started/`: Includes installation and AdalFlow in 10 minutes

`tutorials/`: Includes our main tutorials
`use_cases/`: Includes the use cases of AdalFlow that will be added in the future and which accepts community contributions

`apis/`: All the source-code-related documents will be included in this directory

<!-- `resources/`: Include all the AdalFlow-relevant resources. -->

## How to Edit the Documentation

Most of the documentation updates should be written as comments/doc-strings in your source code, which will be automatically converted to docs. Do manual editing when you add instructions to use your code, adjust the layout, etc.

The existing documentation is a combination of automatic generation and human editing.

### **Source Code Doc-string Update**

The `autodoc` extension in `conf.py` combined with `.. automodule::` in the `.rst` files makes it easy to update documents from the source code.

If you update the existing source code, you only need to run:

```python
cd docs
make clean
make html
```

And your documentation will be updated.

### Add New Code

If you add new modules or code to the project, sphinx has a [command](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html#sphinx-apidoc) to automatically generate the code docs.

```python
sphinx-apidoc [OPTIONS] -o <OUTPUT_PATH> <MODULE_PATH> [EXCLUDE_PATTERN …]
```

***Note:***

If your new module is a folder, it should contain a `__init__.py` file.

Remember to exclude the code that you don’t need in the [EXCLUDE_PATTERN …], otherwise Sphinx will compile them all.

***Example:***

Located in the root directory, run:

```python
sphinx-apidoc -o docs/source/tutorials ./use_cases **test**
```

(*test* is to exclude the files containing `test` in the filename)

You will find a `modules.rst` and a `use_cases.rst`  in the `docs/source/tutorials`. The `use_cases.rst` contains all the packages included in your `./use_cases`.

Then you should add the link to the `index.rst` to show your source code and docs in the documentation. Find `docs/source/index.rst` and add the new section:

```python
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Use Cases

   tutorials/use_cases
```

Then run:

```python
cd docs
make clean
make html
```

And you will be able to find the newly added use_cases module.

### Add New Docs

If you want to add any written files such as README.md to the documentation, there is an easy way to transform the files to `.rst` files using `Pandoc`.

- First, install Pandoc with Homebrew:

    `brew install pandoc`

- Then run `pandoc -s <input .md file> -o <path/to/target_rst_file>`. For example, in the root directory run `pandoc -s README.md -o docs/source/get_started/introduction.rst`.This command will take content from `README.md` and create an `introduction.rst` file in the specified directory.

After editing, run

```python
cd docs
make clean
make html
```

### Commit the Edited Documentation

Remember to exclude any unnecessary files in `.gitignore`. Please don’t commit files in `docs/build`. We can dynamically build local documentation with the make files and `source/`.

Please push your updates to the GitHub repo.

The structure of the code base and the docs:

```
AdalFlow/
├── docs/
│   ├── apis/
│   │   ├── core/
│   │   │   ├── core.module1.rst
│   │   │   ├── core.module2.rst
│   │   ├── components/
│   │   │   ├── components.module1.rst
│   │   │   ├── components.module2.rst
│   ├── build/
│   │   ├── html/
│   │   │   ├── _static/
│   │   │   ├── _templates/
│   │   │   ├── index.html
│   │   │   ├── core/
│   │   │   │   ├── core.module1.html
│   │   │   │   ├── core.module2.html
│   │   │   ├── components/
│   │   │   │   ├── components.module1.html
│   │   │   │   ├── components.module2.html
│   ├── _static/
│   ├── _templates/
│   ├── conf.py
│   ├── index.rst
│   ├── Makefile
├── core/
│   ├── __init__.py
│   ├── module1.py
│   ├── module2.py
├── components/
│   ├── __init__.py
│   ├── module1.py
│   ├── module2.py
```
