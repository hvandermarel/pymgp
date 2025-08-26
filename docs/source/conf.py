# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

sys.path.insert(0, str(Path('..', '..').resolve()))

project = 'pyMGP'
copyright = '2025, Hans van der Marel'
author = 'Hans van der Marel'
release = '0.9'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#    'sphinx.ext.napoleon',
#    'nbsphinx',                 # Jupyter notebook extension
extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'nbsphinx',
]


autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'pydata_sphinx_theme'
#html_theme = 'sphinx_rtd_theme'
#html_theme = 'sphinxdoc'
html_static_path = ['_static']

html_show_sourcelink = False

# autodoc_mock_imports = ["scipy", "datetime", "urllib", "ssl", "dateutil"]