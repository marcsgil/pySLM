# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from pathlib import Path
import sys
from recommonmark.parser import CommonMarkParser

from optics import __version__


sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Optics'
copyright = '2023, Zhaozhong Chen, Braian Pinheiro da Silva, Laurynas Valantinas, Tom Vettenburg'
author = 'Zhaozhong Chen, Braian Pinheiro da Silva, Laurynas Valantinas, Tom Vettenburg'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['m2r2',  # or m2r
              'sphinx.ext.duration',
              'sphinx.ext.doctest',
              'sphinx.ext.autodoc',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon',  # Used to write beautiful docstrings
              'sphinx_autodoc_typehints',  # Used to insert typehints into the final docs
              'added_value',  # Used to embed values from the source code into the docs
              'sphinxcontrib.mermaid',  # Used to build graphs
              'sphinx.ext.intersphinx',
              'sphinx_rtd_theme',
              ]

templates_path = ['_templates']
exclude_patterns = []

autoclass_content = 'class'
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'special-members': True,
    'inherited-members': True,
    'undoc-members': True,
    'exclude-members': '__dict__, __weakref__, __abstractmethods__, __annotations__, __parameters__, __module__, __getitem__, __str__, __repr__, __hash__, ' +
        '__slots__, __orig_bases__, __subclasshook__, __class_getitem__, __contains__, __reversed__, ' +
        '__cause__,  __context__, __delattr__, __getattribute__, __new__, __reduce__, __setattr__, __setstate__, __suppress_context__, __traceback__, ',  # , __eq__, __add__, __sub__, __neg__, __mul__, __imul__, __matmul__, __div__, __idiv__, __rdiv__, __truediv__',
    'show-inheritance': True,
}

napoleon_numpy_docstring = False
todo_include_todos = True

intersphinx_mapping = {'python': ('http://docs.python.org/3', None),
                       'numpy': ('http://docs.scipy.org/doc/numpy/', None),
                       'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
                       'matplotlib': ('http://matplotlib.sourceforge.net/', None),
                       'joblib': ('https://joblib.readthedocs.io/en/latest/', None),
                       }

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'sidebar_collapse': False,
    'show_powered_by': False,
}
html_theme_options = {
    'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your dashboard
    'analytics_anonymize_ip': True,
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'white',
    # ToC options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': -1,
    'includehidden': True,
    'titles_only': False
}
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
# html_sidebars = {
#     '**': [
#         'about.html',
#         'navigation.html',
#         'localtoc.html',
#         'relations.html',
#         'searchbox.html',
#     ],
# }
html_static_path = ['_static']

source_suffix = ['.rst', '.md']

source_parsers = {
    '.md': CommonMarkParser,
}

autodoc_mock_imports = ['torch', 'Lib64', 'nidaqmx', 'ids_peak']
