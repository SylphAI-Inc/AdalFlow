# LightRAG Documentation Contribution Instruction

If you want to contribute to the LightRAG documentation system, please refer to the following instructions.

### Instruction Summary for Quick Navigation

- [Setup](#setup)
- [File Structure](#file-structure)
- [Editing Tips for Sphinx Documentation](#editing-tips-for-sphinx-documentation)

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

Install Sphinx and the required theme directly into your active virtual environment:

`pip install sphinx sphinx-rtd-theme`  install the sphinx package and the sphinx theme

### **3. Build the Documentation**

Navigate to the `docs` directory within your project folder and compile the documentation:

`cd docs`

`make html`

This command generates the documentation from the source files and outputs HTML files to `docs/build/html`.

### **4. View the Documentation**

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

**`get_started/`**

- installation.rst
- introduction.rst

**`tutorials/`**

- simpleQA.rst - This is a dummy file, we should add the tutorials as different.rst files here

**`apis/`**

You can go through the .rst files under `apis` and edit accordingly. Remember to add the unnecessary files into `LightRAG/.gitignore`.

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

Once you've made your edits, rebuild the documentation dynamically to see your changes:

- Clean previous builds:
    
    `make clean`
    
- rebuild HTML documentation:
    
    `make html`
    

### Automatic Update

We have already included the necessary extensions in the configuration(conf.py). Therefore, if you correctly include the source code in `.. automodule::` in the `.rst` file, when you update the code doc string, simply do the rebuilding by `make html`, the documentation will be automatically updated.

For example, `.. automodule:: components.api_client.transformers_client`

Ensure to commit your changes and push them to the GitHub repository to make them available to others. 

### **[Optional] Generate Texts from Doc Strings Automatically**

**You don’t necessarily do this.** But you can use this to quickly generate the text from doc strings.

- You should run the `sphinx-apidoc -o <output_path> <module_path>` to generate the texts. Make sure your module includes __init__.py.
- If you are in the root directory, you can run
    
    `sphinx-apidoc -o docs/source/documents/use_cases use_cases *test*` . `*test*` is to exclude the files containing `test` in the filename 
    
    By doing this, you are generating the code-related texts and pages in the `docs/source/apis`, and the source module path is the current directory `components/` . [sphinx-apidoc command reference](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html).