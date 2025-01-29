# AdalFlow Documentation Guide

## Content Overview

- [AdalFlow Documentation Guide](#adalflow-documentation-guide)
  - [Content Overview](#content-overview)
  - [Introduction](#introduction)
    - [How the Documentation Works](#how-the-documentation-works)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Install Dependencies](#2-install-dependencies)
    - [3. Verify Setup](#3-verify-setup)
  - [File Structure](#file-structure)
    - [**conf.py**](#confpy)
    - [**index.rst**](#indexrst)
    - [**Existing Sections**](#existing-sections)
  - [Editing and Updating Documentation](#editing-and-updating-documentation)
    - [Updating Docstrings in Source Code](#updating-docstrings-in-source-code)
    - [Adding New Code and Docstrings](#adding-new-code-and-docstrings)
  - [Building the Documentation](#building-the-documentation)
    - [Steps to Build](#steps-to-build)
    - [Quick Start](#quick-start)
    - [Viewing the Documentation](#viewing-the-documentation)
  - [Testing the Documentation](#testing-the-documentation)
  - [Contributing](#contributing)
    - [Git Workflow](#git-workflow)
    - [Directory Structure](#directory-structure)
  - [Optional Tools](#optional-tools)
    - [Convert Markdown to reStructuredText](#convert-markdown-to-restructuredtext)

## Introduction

AdalFlow uses [Sphinx](https://www.sphinx-doc.org/en/master/) and [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) for generating project documentation. Sphinx reads configurations from a Python script (`conf.py`), pulls documentation from comments in the code (via the `autodoc` extension), and organizes content through `.rst` files.

### How the Documentation Works

- **Configuration**: Managed via `conf.py`.
- **Automatic Documentation**: Generated from docstrings using `autodoc`.
- **Content Organization**: Structured through `.rst` files and the table of contents defined in `index.rst`.

This guide will walk you through setting up, building, and contributing to the AdalFlow documentation.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python**: Version 3.8 or higher
- **Poetry**: For dependency management. Install it with:

  ```bash
  pip install poetry
  ```

- **Optional Tools**:
  - `pandoc` for converting Markdown to reStructuredText.
  - A modern browser for viewing documentation.

## Setup

### 1. Clone the Repository

Clone the AdalFlow GitHub repository:

```bash
git clone https://github.com/SylphAI-Inc/AdalFlow.git
cd AdalFlow
```

### 2. Install Dependencies

Install all necessary dependencies using `poetry`:

```bash
poetry install
```

If you encounter issues with `poetry`, ensure it is up-to-date by reinstalling:

```bash
pip install --upgrade poetry
```

**Note:** Be sure to run these commands from the root level of the project.

**Alternative: Using pip**

For users who prefer not to use Poetry, you can install dependencies using `pip` by ensuring a `requirements.txt` file is present:

```bash
pip install -r requirements.txt
```

*Ensure that `requirements.txt` is kept up-to-date with necessary dependencies.*

### 3. Verify Setup

Ensure all dependencies are installed correctly:

```bash
poetry check
```

## File Structure

### **conf.py**

The `docs/source/conf.py` controls the configurations used by Sphinx to build the documentation, including the project-related information, [sphinx extensions](https://www.sphinx-doc.org/en/master/usage/extensions/index.html), templates configuration, HTML theme, patterns to exclude, language configuration, project path setup, etc.

### **index.rst**

The `docs/source/index.rst` is the root document for Sphinx-generated documentation ("homepage" for the documentation site). It includes the `toctree` that defines the documentation hierarchical structure (sections/chapters). It also links to other `.rst` files that users can navigate through.

For example, in the `index.rst`, the `:caption: Get Started` corresponds to the section name of the documentation site. `installation` and `introduction` are the detailed pages.

```rst
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Get Started

   get_started/installation
   get_started/introduction
```

### **Existing Sections**

Existing sections include:

- `get_started/`: Includes installation and AdalFlow in 10 minutes
- `tutorials/`: Includes our main tutorials
- `use_cases/`: Includes the use cases of AdalFlow that will be added in the future and which accepts community contributions
- `apis/`: All the source-code-related documents will be included in this directory

## Editing and Updating Documentation

### Updating Docstrings in Source Code

Most of the documentation is automatically generated from code comments using the `autodoc` extension. To update:

1. **Edit the docstrings** in your source code.
2. **Rebuild the documentation** (see [Building the Documentation](#building-the-documentation)).

**Note:** Ensure that your docstrings follow the [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) format for proper rendering.

### Adding New Code and Docstrings

When you add **new** modules or code to the project, it's essential to generate corresponding `.rst` files to include them in the documentation. Here's how to do it:

1. **Ensure Proper Structure**:
   - If your new module is a folder, it should contain an `__init__.py` file to be recognized as a package.
   - Write comprehensive docstrings for your new code to leverage the `autodoc` extension effectively.

2. **Generate `.rst` Files Using `sphinx-apidoc`**:

   Use the `sphinx-apidoc` command to automatically generate `.rst` files for your new modules:

   ```bash
   sphinx-apidoc [OPTIONS] -o <OUTPUT_PATH> <MODULE_PATH> [EXCLUDE_PATTERN …]
   ```

   **Example:**

   While located in the root directory of your project, run the following command to generate `.rst` files for the `use_cases` module, excluding any files or directories with "test" in their names:

   ```bash
   sphinx-apidoc --force -o docs/source/tutorials ./use_cases "*test*"
   ```

   - **`--force`**: Overwrites any existing `.rst` files in the output directory. This is safe as long as you haven't made manual edits to these `.rst` files.
   - **`-o docs/source/tutorials`**: Specifies the directory where the `.rst` files will be generated.
   - **`./use_cases`**: Path to the module or package to document.
   - **`"*test*"`**: Excludes any files or directories with "test" in their names.

   **Caution:** The `--force` flag will overwrite existing `.rst` files in the specified output directory. Ensure that you haven't made manual edits to these files that you wish to preserve. If you have custom `.rst` files, consider backing them up or omitting the `--force` flag.

3. **Update `index.rst` to Include New Sections**:

   After generating the `.rst` files, update the `index.rst` to include the new documentation sections. For example:

   ```rst
   .. toctree::
      :glob:
      :maxdepth: 1
      :caption: Tutorials

      tutorials/use_cases
   ```

   This ensures the new module is linked in the documentation's table of contents.

4. **Rebuild the Documentation**:

   After adding new docstrings and generating `.rst` files, [rebuild](#building-the-documentation) the documentation to see the updates

   **Summary of Steps When Adding New Code with Docstrings:**

   1. **Add the new code/module** with proper docstrings.
   2. **Generate `.rst` files** using `sphinx-apidoc`.
   3. **Update `index.rst`** to include new sections.
   4. **Rebuild the documentation** to reflect changes.


## Building the Documentation

### Steps to Build

1. **Navigate to the `docs` Directory**:

   ```bash
   cd docs
   ```

2. **Clean Previous Builds**:

   Remove previous build artifacts to ensure a fresh build.

   ```bash
   make clean
   ```

3. **Build the Documentation**:

   Generate the HTML documentation.

   ```bash
   make html
   ```

4. **(Optional) Build with Verbose Output**:

   For more detailed build logs, useful for debugging.

   ```bash
   sphinx-build -b html source build -v
   ```

### Quick Start

For users who prefer straightforward commands without additional options:

```bash
cd docs
make html
```

After the build completes, open the documentation:

- **macOS**:

  ```bash
  open build/html/index.html
  ```

- **Linux**:

  ```bash
  xdg-open build/html/index.html
  ```

- **Windows**:

  Manually open the `index.html` file in your preferred browser.

### Viewing the Documentation

After building, the HTML files will be available in `docs/build/html`. Open `index.html` in your browser to view the documentation.

**Handling Browser Restrictions on Local Resources:**

Some browsers restrict loading local resources like CSS for security reasons. In such cases, use a local server to serve the files:

```bash
cd docs/build/html
python -m http.server
```

Visit `http://127.0.0.1:8000/` in your browser to view the documentation.

## Testing the Documentation

To ensure the documentation builds correctly without errors or warnings, run:

```bash
sphinx-build -n -W --keep-going source build
```

- **`-n`**: Runs Sphinx in nit-picky mode to catch missing references.
- **`-W`**: Treats warnings as errors, causing the build to fail if any warnings are present.
- **`--keep-going`**: Continues the build as much as possible after encountering errors.

**Recommendation:** Fix any issues reported by this command before committing your changes to maintain documentation quality.

## Contributing

### Git Workflow

To maintain a clean repository, exclude unnecessary files from commits. Specifically, avoid committing the `docs/build` directory, as documentation builds are dynamic and can be regenerated locally.

**Example `.gitignore`:**

```plaintext
# Ignore build files
docs/build/
*.pyc
__pycache__/
```

- **Commit Only Necessary Files**:
  - Source files (`.rst`, `.py`, etc.)
  - Configuration files (`conf.py`, `Makefile`, `pyproject.toml`, etc.)
- **Exclude**:
  - `docs/build/`: Generated HTML files.
  - Compiled Python files (`*.pyc`, `__pycache__/`).

### Directory Structure

Ensure the project follows this structure:

```plaintext
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
│   ├── pyproject.toml
│   ├── poetry.lock
```

**Note:** The `build/` directory contains generated files and should not be manually edited or committed.

## Optional Tools

If you want to add any written files such as `README.md` to the documentation, there is an easy way to transform the files to `.rst` files using `Pandoc`.

### Convert Markdown to reStructuredText

To convert `.md` files (e.g., `README.md`) to `.rst`:

1. **Install `pandoc`**:

   ```bash
   brew install pandoc
   ```

   *For non-Homebrew users, refer to [Pandoc's installation guide](https://pandoc.org/installing.html).*

2. **Run the Conversion**:

   **Conversion Syntax:**

   ```bash
   pandoc -s <input.md> -o <path/to/target.rst>
   ```

   **Example:**

   This command will take content from `README.md` and create an `introduction.rst` file in the specified directory.

   ```bash
   pandoc -s README.md -o docs/source/get_started/introduction.rst
   ```

3. **Rebuild the Documentation**:

   After converting, rebuild the documentation to include the new `.rst` files.

   ```bash
   cd docs
   make clean
   make html
   ```

**Note:** Ensure that the converted `.rst` files are correctly linked in your `index.rst` or appropriate parent `.rst` files.
