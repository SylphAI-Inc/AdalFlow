import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../lightrag/lightrag"))

project = "LightRAG"
copyright = "2024, SylphAI, Inc"
author = "SylphAI, Inc"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
    "sphinx_design",
    "sphinx_copybutton",
    "nbsphinx",
    "sphinx_search.extension",
    # "sphinx_sitemap",
]

html_show_sphinx = False

templates_path = ["_templates"]

exclude_patterns = ["lightrag/tests", "test_*", "../li_test"]

html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False

html_logo = "./_static/images/LightRAG-logo-doc.jpeg"

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 8,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/SylphAI-Inc/LightRAG",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Discord",
            "url": "https://discord.gg/ezzszrRZvT",
            "icon": "fa-brands fa-discord",
        },
    ],
    "navbar_end": ["navbar-icon-links.html", "search-field.html"],
}

html_static_path = ["_static"]

html_short_title = "LightRAG"
html_favicon = "./_static/images/LightRAG-logo-circle.png"
html_theme_options = {
   "announcement": "⭐️ If you find LightRAG helpful, give it a star on <a href='https://github.com/SylphAI-Inc/LightRAG'> GitHub! </a> ⭐️",
}

# html_meta = {
#     "description": "The Lightning Library for LLM Applications",
#     "keywords": "LLM, Large language models, nlp, agent, machine-learning framework, ai, chatbot, rag, generative-ai",
# }

autosummary_generate = False
autosummary_imported_members = False
add_module_names = False
autosectionlabel_prefix_document = True
autodoc_docstring_signature = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "member-order": "bysource",
    "show-inheritance": True,
    "private-members": False,
    "inherited-members": False,
    "exclude-members": "__init__",
}


def setup(app):
    app.add_css_file("css/custom.css")
