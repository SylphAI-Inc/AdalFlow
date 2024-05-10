# LightRAG Documentation Contribution Instruction

If you want to contribute to the LightRAG documentation system, please refer to the following instructions.

## Table of Contents
1. [Setup](#setup)
   - [Prerequisites](#prerequisites)
   - [Clone the GitHub Project](#clone-the-github-project)
   - [Install Necessary Packages](#install-necessary-packages)
   - [Optional: Generate Texts from Doc Strings](#optional-generate-texts-from-doc-strings)
   - [Build the Documentation](#build-the-documentation)
   - [View the Documentation](#view-the-documentation)
2. [File Structure](#file-structure)
   - [conf.py](#confpy)
   - [index.rst](#indexrst)
   - [Sections](#sections)
3. [Editing Tips for Sphinx Documentation](#editing-tips-for-sphinx-documentation)
   - [Directly Edit an Existing .rst File](#directly-edit-an-existing-rst-file)
   - [Create a New .rst File](#create-a-new-rst-file)
   - [Convert a Markdown File to .rst Using Pandoc](#convert-a-markdown-file-to-rst-using-pandoc)

## Setup

### **Prerequisites**

Before you start, please ensure you have:

- **Basic knowledge of command-line operations and Git:** These tools are essential for managing our project files. If you're new to these, check out [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics) and Command Line Basics.
- **LightRAG project set up on your machine:** This ensures you can run and test changes locally.

### **1. Clone the Github Project**

Clone the repository to get the latest source files. The HTML files are not included in the repository because they are generated dynamically and can consume considerable space.

Please run the following command:

`git clone https://github.com/SylphAI-Inc/LightRAG.git`

### **2. Install Necessary Packages**

Install Sphinx and the required Sphinx theme directly into your active virtual environment:

`pip install sphinx sphinx-rtd-theme`  install the sphinx package and the sphinx theme

**[Optional]** For contributors planning to convert Markdown documents to **`.rst`**,  install Pandoc, a package to transform the README.md file to `.rst` files. Run

 `brew install pandoc`

### **3. [Optional in setup] Generate Texts from Doc Strings**

**You don’t necessarily do this in setup.** But you should better run the following instructions if you have modified the codebase.

- To generate texts from the doc strings and show source code conveniently, you should firstly find the `docs/source/conf.py` . In this file, make sure the extensions `'sphinx.ext.autodoc’, 'sphinx.ext.viewcode'`  are included.
- Then you should run the `sphinx-apidoc -o <output_path> <module_path>` to generate the texts. In LightRAG, we have multiple modules and subdirectories. Therefore, if you are in the root directory, you can run
    
    `sphinx-apidoc -o docs/source/documents . -f -e` 
    
    By doing this, you are generating the code-related texts and pages in the `docs/source/documents`, and the module path is the current directory`.` containing `components/` and `core/` . [sphinx-apidoc command reference](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html).
    

### **4. Build the Documentation**

Navigate to the `docs` directory within your project folder and compile the documentation:

`cd docs`

`make html`

This command generates the documentation from the source files and outputs HTML files to `docs/build/html`.

### **5. View the Documentation**

After building the documentation, you can view it by opening the `index.html` file located in `docs/build/html`. You can open this file in any web browser to review the generated documentation.

## File Structure

The files to edit are located in `docs/source`. 

### **conf.py**

This file contains the configurations, including the paths setup, Project information, and General configuration(extension, templates, HTML theme, exclude patterns)

### **index.rst**

This file contains the information on the home(index) page. You can edit the sections that appear on this page by modifying the toctree modules.

Each toctree can represent a section, when you update the caption, you will update the names of the sections. For example, :caption: Get Started

You will see that at the end of the toctree, there are paths linked, such as `get_started/installation`. This path points to the installation.rst file in docs/source/get_started, which contains the information to show in the installation section.

### **Sections**

The documentation has 

**`get_started/`**

- installation.rst
- introduction.rst

**`tutorials/`**

- simpleQA.rst - This is a dummy file, we should add the tutorials as different.rst files here

**`documents/`**

The files in `documents` are the code APIs to present. Please refer to Setup Generate Texts from Doc Strings[Optional in setup] to generate the texts and source code.

You can go through the .rst files under `documents` and edit accordingly. Remember to add the unnecessary files into `LightRAG/.gitignore`.

**`Resources/`**

This folder contains a `resource.rst` file with the necessary links and sources to help the developers.

## Editing Tips for Sphinx Documentation

To effectively edit the LightRAG documentation, you have several options depending on your specific needs:

### 1. Directly Edit an Existing .rst File

Locate the `.rst` file within the `docs/source` directory that you wish to edit. You can modify the content directly in any text editor. For formatting help, refer to the reStructuredText Quickstart Guide:
- [Quickstart](https://docutils.sourceforge.io/docs/user/rst/quickstart.html)
- [reStructuredText Markup Specification](https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html)

### 2. Create a New .rst File

If you need to add a new section or topic:

- Create a new `.rst` file in the appropriate subdirectory within `docs/source`.
- Write your content following reStructuredText syntax.
- If you are creating a new section, ensure to include your new file in the relevant `toctree` located usually in `index.rst` or within the closest parent `.rst` file, to make it appear in the compiled documentation.

### 3. Convert a Markdown File to .rst Using Pandoc

To integrate content written in Markdown into the Sphinx project, use Pandoc to convert it to `.rst` format:

Pandoc is a package to transform the files to `.rst` files.

- First, install Pandoc with Homebrew:
    
    `brew install pandoc` 
    
- You might also want to combine the [sphinx extensions](https://www.sphinx-doc.org/en/master/usage/extensions/index.html) in your `doc/source/conf.py` for a better layout.
- Then run `pandoc -s <input .md file> -o <path/to/target_rst_file>` . For example, in the root directory `pandoc -s README.md -o docs/source/get_started/introduction.rst` .This command will take content from `README.md` and create an `introduction.rst` file in the specified directory.

### After editing

Once you've made your edits, rebuild the documentation to see your changes:

- Clean previous builds:
    
    `make clean`
    
- Generate new HTML documentation:
    
    `make html`
    

Ensure to commit your changes and push them to the GitHub repository to make them available to others.