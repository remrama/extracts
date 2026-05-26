"""Sphinx configuration for the extracts docs."""

import os

import extracts

project = "extracts"
release = extracts.__version__
version = extracts.__version__
author = "Remington Mallett"
copyright = f"2024-%Y, {author}"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "jupyter_execute", ".jupyter_cache"]

# Source files: .md goes through myst-nb (which extends myst-parser); .rst stays default.
source_suffix = {
    ".md": "myst-nb",
    ".rst": "restructuredtext",
}

# -- MyST / MyST-NB ----------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist", "substitution"]

# Build-time execution of fenced ``{code-cell}`` blocks.
nb_execution_mode = "cache"
nb_execution_timeout = 180
nb_execution_raise_on_error = True
nb_execution_cache_path = os.path.join(os.path.dirname(__file__), ".jupyter_cache")

# -- Autodoc / Napoleon ------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
python_use_unqualified_type_names = True

napoleon_google_docstring = False

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pooch": ("https://www.fatiando.org/pooch/latest/", None),
}

# -- HTML output -------------------------------------------------------------

html_title = project
html_copy_source = False
html_show_copyright = False
html_show_sphinx = False
html_use_index = False
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": f"https://github.com/remrama/{project}",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": False,
    "use_issues_button": False,
    "use_download_button": False,
    "home_page_in_toc": False,
    "show_toc_level": 2,
    "toc_title": "On this page",
    "pygments_light_style": "github-light-colorblind",
    "pygments_dark_style": "github-dark-colorblind",
}

templates_path = ["_templates"]
source_encoding = "utf-8"
add_function_parentheses = False
