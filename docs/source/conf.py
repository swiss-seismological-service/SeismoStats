# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil

import setuptools_scm

# Path to the project's root CHANGELOG.md
changelog_src = os.path.abspath("../../CHANGELOG.md")
changelog_dst = os.path.abspath("./changelog.md")

# Copy CHANGELOG.md into the docs folder
if os.path.exists(changelog_src):
    shutil.copy(changelog_src, changelog_dst)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SeismoStats'
copyright = '2024, multiple'
author = 'multiple'
version = setuptools_scm.get_version(root='../..',
                                     relative_to=__file__)
release = setuptools_scm.get_version(root='../..',
                                     relative_to=__file__,
                                     version_scheme="python-simplified-semver",
                                     local_scheme="no-local-version")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.autosectionlabel',
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'nbsphinx_link'
]

autosectionlabel_prefix_document = True

nbsphinx_custom_formats = {}
templates_path = ['_templates']
exclude_patterns = []
napoleon_custom_sections = [('Returns', 'params_style')]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {
    "use_edit_page_button": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/swiss-seismological-service/SeismoStats",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ]
}
html_show_sourcelink = False
navigation_with_keys = True
html_context = {
    "default_mode": "light",
}
suppress_warnings = ['autosectionlabel.*',
                     'myst.header',
                     'config.cache']
