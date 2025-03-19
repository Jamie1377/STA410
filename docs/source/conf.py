# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'STA410-Stock-Prediction'
copyright = '2025, Jamie Yu'
author = 'Jamie Yu'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []
extensions = [
    'sphinx.ext.autodoc',    # Auto-generate docs from docstrings
    'sphinx.ext.napoleon',   # Support Google/NumPy-style docstrings
    'sphinx.ext.viewcode',   # Add links to source code
    'sphinx.ext.intersphinx' # Link to external docs (e.g., NumPy)
]

# Add your package to the Python path
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Adjust path to your package

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "github_url": "hhttps://github.com/Jamie1377/STA410",
    "logo": {
        "image_light": "logo.png",
        "image_dark": "logo-dark.png",
    }
}