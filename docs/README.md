## Steps - sphinx + github + python + read the doc

**If you want to set up the documentation for the github project from scratch:**

1. git clone repo
2. create a `docs/` folder under the project
3. install sphinx in that `docs/` (virtual environment)
4. `sphinx-quickstart` for configuration
5. edit the `index.rst` - home page of the documentation
6. add the extensions to the conf.py
    
    extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'autodocsumm',
    ]
    
7. modify the path, insert the system path
8. to generate the APIs descriptions, in the project root directory(LightRAG/), run `sphinx-apidoc o -docs .` , this helps create the function introductions from the doc strings
    
    `usage: sphinx-apidoc [OPTIONS] -o <OUTPUT_PATH> <MODULE_PATH> [EXCLUDE_PATTERN, ...]`, 
    
    If the project contains **multiple sub directories**, run `sphinx-apidoc -o docs . ..`  to consider subdirectories
    
9. build in the doc folder `make html`
10. use in-built themes, change in `conf.py` in **html_theme**, we can use `pip install sphinx-rtd-theme` to use the same theme as PyPI documentation. and later do more customization
11. markers to use: https://docutils.sourceforge.io/docs/user/rst/quickstart.html, https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html. We need to be consistent with this documentation. And we are able to put images as well.
12. edit the documents, push the files under `docs/` to the repo
13. use [read the docs](https://about.readthedocs.com/?ref=readthedocs.com), create an account, import the project repo link and build it

**If you already have the documentation set up:**

1. clone the github repo
2. go to the documentation directory, e.g. `docs/`
3. edit the files(.rst)
4. build with `make html` to update the pages
5. push the updates
6. probably need to update on [read the docs](https://about.readthedocs.com/?ref=readthedocs.com)