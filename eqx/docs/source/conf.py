import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

project = "EquivariantX"
copyright = "2026, Zemin Xu"
author = "Zemin Xu"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

autodoc_member_order = "bysource"
autodoc_typehints = "none"
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_end": ["theme-switcher"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/xvzemin/tace",
            "icon": "fab fa-github",
        }
    ],
    "show_nav_level": 2,
}
html_show_sourcelink = False
