import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../adalflow/adalflow"))

project = "AdalFlow"
copyright = "2025, SylphAI, Inc"
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
    "sphinx_tabs.tabs",
    "sphinx_design",
    "sphinx_copybutton",
    "myst_parser",  # Enable markdown support
    "sphinxcontrib.mermaid",  # Enable Mermaid charts
    # "sphinx_sitemap",
]

html_show_sphinx = False

templates_path = ["_templates"]

exclude_patterns = ["adalflow/tests"]

html_theme = "furo"

html_show_sourcelink = False

html_static_path = ["_static"]

# autodoc_mock_imports = ["datasets"]

html_theme_options = {
    "light_logo": "images/AdalFlow.svg",  # For light mode
    "dark_logo": "images/AdalFlow_black_bg.svg",  # For dark mode
    "top_of_page_button": "edit",  # Remove default Furo buttons
    "light_css_variables": {
        "color-brand-primary": "#FF6F00",
        "color-brand-content": "#1E2A38",
    },
    "dark_css_variables": {
        "color-brand-primary": "#FF8F00",
        "color-brand-content": "#CFD8DC",
    },
    "dark_mode_code_blocks": False,  # Remove mixed code block styling
    "theme_switcher": ["light", "dark"],  # Show only two theme options
    "collapse_navigation": False,
    "navigation_depth": 8,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/SylphAI-Inc/AdalFlow",
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
    #                 <p>⭐️ If you find AdalFlow helpful, please star us on <a href='https://github.com/SylphAI-Inc/AdalFlow'>GitHub!</a> ⭐️</p>
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


html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "github-link.html",  # New custom link
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}

html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
]

html_title = "Build and Optimize LM Workflows"
# html_title = "AdalFlow: The Library to Build and Auto-Optimize LLM Task Pipelines"
html_favicon = "./_static/images/LightRAG-logo-circle.png"
# html_context = {
#     "docstitle": "AdalFlow: The Lightning Library for LLM Applications"
# }
# html_theme_options = {
#    "announcement": "⭐️ If you find AdalFlow helpful, give it a star on <a href='https://github.com/SylphAI-Inc/AdalFlow'> GitHub! </a> ⭐️",
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

# MyST parser configuration for markdown files
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Configure MyST to handle Mermaid code blocks
myst_fence_as_directive = ["mermaid"]

# Mermaid configuration
mermaid_version = "latest"  # Use latest version of Mermaid
mermaid_init_js = "mermaid.initialize({startOnLoad:true});"

# Include markdown files in source parsing
source_suffix = {
    ".rst": None,
    ".md": None,
}

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


from unittest import mock

try:
    import datasets as hf_datasets
except ImportError:
    hf_datasets = mock.Mock()
    sys.modules["hf_datasets"] = hf_datasets
