project = "TACE"
copyright = "2026, xuzemin"
author = "xuzemin"
release = "v0.2.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx.ext.autosummary",
    "myst_parser",
]

myst_enable_extensions = [
    "html_admonition",
    "dollarmath",
    "amsmath",
]

myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_typehints = "none"
autosummary_generate = True
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
templates_path = ["_templates"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "e3nn": ("https://docs.e3nn.org/en/stable/", None),
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    # "logo": {
    #     "text": "TACE",
    # },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    # "navbar_end": ["search-field.html", "theme-switcher"],
    "navbar_end": ["theme-switcher"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/xvzemin/tace",
            "icon": "fab fa-github",
        },
    ],
    "show_nav_level": 1,
    "footer_start": [],
    "footer_end": [],
}
html_show_sourcelink = False
html_static_path = ["_static"]
# html_logo = "_static/logo.png"
html_favicon = "_static/favicon.svg"

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
