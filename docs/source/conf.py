import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../adalflow/adalflow"))

project = "AdalFlow"
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

exclude_patterns = ["adalflow/tests"]

html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False

html_logo = "./_static/images/adalflow-logo.png"

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
    # "announcement": """
    # <script>
    #     document.addEventListener("DOMContentLoaded", function() {
    #         var bannerClosed = localStorage.getItem("bannerClosed");
    #         var bannerHeader = document.querySelector('.bd-header-announcement');
    #         if (bannerClosed !== "true") {
    #             // Create the announcement banner div dynamically
    #             var banner = document.createElement('div');
    #             banner.id = 'announcement-banner';
    #             banner.className = 'announcement-banner';
    #             // Create the content for the announcement
    #             banner.innerHTML = `
    #                 <p>⭐️ If you find LightRAG helpful, give it a star on <a href='https://github.com/SylphAI-Inc/LightRAG'>GitHub!</a> ⭐️</p>
    #                 <button onclick="closeBanner()">×</button>
    #             `;
    #             // Append the banner to the banner header
    #             if (bannerHeader) {
    #                 bannerHeader.querySelector('.bd-header-announcement__content').appendChild(banner);
    #             }
    #             // Function to close the banner and remove it from the DOM
    #             window.closeBanner = function() {
    #                 if (bannerHeader) {
    #                     bannerHeader.parentNode.removeChild(bannerHeader);
    #                 }
    #                 localStorage.setItem('bannerClosed', 'true');
    #             };
    #             // Scroll listener to hide banner on scroll
    #             window.addEventListener('scroll', function() {
    #                 if (document.documentElement.scrollTop > 0) {
    #                     banner.style.display = 'none';
    #                 } else {
    #                     banner.style.display = 'flex'; // Ensure it remains flex
    #                 }
    #             });
    #         } else {
    #             if (bannerHeader) {
    #                 bannerHeader.parentNode.removeChild(bannerHeader);
    #             }
    #         }
    #     });
    # </script>
    # """,
    "navbar_end": [
        "navbar-icon-links.html",
        "search-field.html",
    ],  # Add search field here
    "search_bar_text": "Search...",
}


html_static_path = ["_static"]

# html_short_title = "Lightrag"
html_title = "LightRAG: The Lightning Library for LLM Applications"
html_favicon = "./_static/images/LightRAG-logo-circle.png"
# html_context = {
#     "docstitle": "LightRAG: The Lightning Library for LLM Applications"
# }
# html_theme_options = {
#    "announcement": "⭐️ If you find LightRAG helpful, give it a star on <a href='https://github.com/SylphAI-Inc/LightRAG'> GitHub! </a> ⭐️",
# }

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
