project = "scarab"
html_theme = "furo"
author = "Ujjwal Panda"
master_doc = "contents"
exclude_patterns = [""]
copyright = "2021-2022, Ujjwal Panda"
html_baseurl = "https://scarab.readthedocs.io"

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

extensions = extensions = [
    "nbsphinx",
    "myst_parser",
    "sphinx_sitemap",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosectionlabel",
]
