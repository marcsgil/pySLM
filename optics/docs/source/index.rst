.. Optics documentation master file, created by
   sphinx-quickstart on Wed Mar 15 15:05:01 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================
The Optics Library
==================

Overview
________
This library contains the packages ``optics``, ``examples``, ``projects``, and ``tests``.

* :doc:`General overview of the API <./api/modules>`
   - :doc:`optics.utils <./api/optics.utils>` General utility modules.
   - :doc:`optics.calc <./api/optics.calc>` Computational tools.
   - :doc:`optics.instruments <./api/optics.instruments>` Interfaces to instruments.
   - :doc:`optics.gui <./api/optics.gui>` Graphical user interfaces.
   - ``experimental`` A staging space for work in progress before integrating it in the categories above. Code, functions, classes, and submodules that are far from finishes. Some of these may later be migrated to the main submodules of the ``optics`` library.
   - ``external`` Third-party code, not readily available through the PyPI package index.
* ``examples`` Example scripts of how to use the library.
* ``projects`` Project-specific scripts and functions, unlikely of interest to other projects.
* ``tests`` Automated tests of the library's functionality. Useful when modifying widely-used functionality

The documentation for this project is in the ``docs`` folder. Running the ``setup.py`` script in the main folder
generates a website using `Sphinx <https://www.sphinx-doc.org/>`_,
Its output is found in ``docs/build/html/index.html``, and is automatically opened by the ``setup.py`` script.

.. mdinclude:: ../../README.md


==================
Complete Structure
==================

.. toctree::
   :maxdepth: 8

   api/modules
   genindex
   modindex