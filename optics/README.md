# Optics Package for the Lab

## Introduction
This Python 3 package contains code used for optical microscopy experiments at the University of Dundee.

**[MIT License](https://opensource.org/licenses/MIT): https://opensource.org/licenses/MIT**

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/macromax)](https://www.python.org/downloads)
[![PyPI - License](https://img.shields.io/pypi/l/macromax)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/macromax?label=version&color=808000)](https://github.com/tttom/MacroMax/tree/master/python)
[![PyPI - Status](https://img.shields.io/pypi/status/macromax)](https://pypi.org/project/macromax/tree/master/python)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/macromax?label=python%20wheel)](https://pypi.org/project/macromax/#files)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/macromax)](https://pypi.org/project/macromax/)
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/tttom/MacroMax)](https://github.com/tttom/MacroMax)
[![GitHub last commit](https://img.shields.io/github/last-commit/tttom/MacroMax)](https://github.com/tttom/MacroMax)
[![Libraries.io dependency status for latest release](https://img.shields.io/librariesio/release/pypi/macromax)](https://libraries.io/pypi/macromax)
[![Documentation Status](https://readthedocs.org/projects/macromax/badge/?version=latest)](https://readthedocs.org/projects/macromax)

## Getting started
### Download Python
Make sure that you have a recent version of Python installed. The library is tested on Python 3.7 and [newer versions](https://www.python.org/downloads/). 
The latest stable version of Python can be downloaded from: https://www.python.org/downloads/

This library further requires a number of Python packages (as listed in `lab/code/python/optics/requirements.txt`). Python comes with the `pip` tool to download additional packages such as `numpy` and `scipy`. Integrated Development Environments as [PyCharm](https://www.jetbrains.com/pycharm/download/) or [Intellij IDEA](https://www.jetbrains.com/idea/download/) can install these automatically.  

### Get the library
It is most practical to clone the library's repository using GIT. This ensures that you always have the latest versions of the library and simplifies modifications to it. 
Clone the repository to a local folder using either [TortoiseGit](https://tortoisegit.org/) on Windows or a similar tool:
```sh
git clone https://github.com/tttom/lab
```
Note that the installation of [TortoiseGit](https://tortoisegit.org/) can usually be done using the default options. As part of its installation, you will likely also be prompted to install Git for Windows, which can also be done using the default options. Once installed, you can right-click on the folder (e.g. `Documents`) where you want to download the library, and select `Git Clone`. As `URL`, specify: https://github.com/tttom/lab and use your GitHub username and password to clone the repository.
**Do not choose a shared drive** as OneDrive, DropBox, or Google Drive. Sooner or leater this will cause synchronization issues that corrupt the repository.
The latest version should now be downloaded into the `lab` folder. To later get updates, just do a
`git pull` in the `lab` folder. More information on [TortoiseGit](https://tortoisegit.org/docs/tortoisegit/tgit-dug.html) and [GIT](https://guides.github.com/introduction/git-handbook/) in general can be found [here](https://tortoisegit.org/docs/tortoisegit/tgit-dug.html) and [here](https://guides.github.com/introduction/git-handbook/). 

### Run the project
The project can be opened with e.g. the [PyCharm](https://www.jetbrains.com/pycharm/download/) or [Intellij IDEA](https://www.jetbrains.com/idea/download/) Community Edition programming environments. Make sure you have a version of Python 3, not version 2. Preferably get Python 3.7 or later. Start the development environment (e.g. PyCharm or Intellij IDEA) and open the `optics` project in the `lab/code/python/` directory. Once the project is opened, make sure that the development environment has located your Python interpreter. In `File > Project Structure...` set `Project Settings > Project > Project SDK` to the `Python SDK` System Interpreter you have installed, or set up a new Virtual Environment for this project.     

Most likely some Python packages will require installation. PyCharm or IntelliJ IDEA will prompt you to download and install these. This may take a will take a while. Once all requirements are installed, navigate to `optics/examples` and run `first.py` by right-clicking and selecting run (or hitting CTRL-SHIFT-F10). This should output:
```sh
OK, numpy is loading! 
```
You should see further information if the logs are working and `optics` package is loading too:
```sh
...|optics-INFO: Great, also the logs are working!
...|optics-INFO: Created the region-of-interest Roi(top_left=[0 0], shape=[100 100], dtype=int32)
```
The logs are time-stamped and have an indicator of severity: DEBUG, INFO, WARN, ERROR, and FATAL. All messages, including DEBUG messages can also be found in `optics.log`. The logs are configured in `/setup.cfg`, more information can be found  [here](https://docs.python.org/3/library/logging.html).

### Prerequisites

This library requires Python 3 with the modules `numpy` and `scipy` for the main calculations.
From the main library, the modules `sys`, `io`, `os`, and `multiprocessing` are imported; as well as the modules `logging` and `time` for diagnostics.
The `pyfftw` module can help speed up the calculations.

The examples require `matplotlib` for displaying the results.
The `pypandoc` module is required for translating this document to other formats.

The code has been tested on Python 3.7.

## Introduction to the library

### Code organization
The `optics` project contains one main Python package, `optics`, the `examples` package, and a test package, `tests`.

The `optics` package contains the subpackages:
* `calc`: Some optics calculation code (very limited at the moment). Here we should place point-spread function calculations etc.
* `experimental`: A space for code that may or may not become permanent.
* `external`: Third party code, with perhaps a different license.
* `gui`: Graphical user interfaces. At the moment this only contains a draft for the monitor gui.
* `instruments`: Classes to control laboratory instruments. This has subfolders for every device type: `Cam`, `SLM`, and `DM`. New packages will be added for stages, light sources, and galvos.
* `utils`: General functions. It also contains further subpackages for array manipulation and display. Here we should also place the Zernike polynomial functions.

The `examples` package contains short examples of how to use the library.
The `tests` package contains automated code unit tests.

Other files:
* `requirements.txt`: Contains the packages required for the code to run (and perhaps some redundant packages at the moment).
* `setup.cfg`, `setup.py`, `MANIFEST.in`,  and `LICENSE.txt`: These files are only required to make the package publicly available.
* `.gitnore`: A file telling `git` what should not be committed to the repository (e.g. log files).
* `optics.iml` and `.idea` folder: Intellij IDEA project files.
* `__init___.py`: If a folder contains this file, it makes it a Python package. It can contain code to be run when the package is imported. In this case, it defines the `log` object, which determines what will happen to the message output.

#### A simple example
```python
from optics.utils import Roi

r = Roi(center=(50, 50), shape=(100, 100))
print(r)
```
should output:
`Roi(top_left=(0,0), shape=(100,100), dtype=int32)`
when typed on a Python console. In Intellij you can open the Python console from Tools menu.

More examples can be found in the `examples` directory.

#### Running the monitor gui
```python
# Import the graphical user interface
import optics.gui.monitor as mon

# Run it
mon.start()
```

### Contributing to the library

#### Documentation files
Please keep these and other documentation files up to date. The syntax used is [markdown](https://www.markdownguide.org/basic-syntax).

Source code also contains some documentation, though it is far from complete. In particular outward facing interfaces should have function or class documentation between """triple quotes""", describing the arguments used.

#### Source code
Naming is important. It avoids ambiguity, and makes finding things easier. Package, class, function, and variable names should be consistent, descriptive, and adhere to the [Python style guidelines](https://www.python.org/dev/peps/pep-0008/). Overviews can be found in:
* [https://namingconvention.org/python/](https://namingconvention.org/python/)
* [https://docs.python-guide.org/writing/style/](https://docs.python-guide.org/writing/style/)
* [https://google.github.io/styleguide/pyguide.html](https://google.github.io/styleguide/pyguide.html)
A good IDE will highlight most style errors and facilitate changing names across all files in the project. 

Python has different ways of achieving the same thing. Following best practices helps keeping everything clear. Some guides can be found [here](https://www.jeffknupp.com/blog/2012/10/04/writing-idiomatic-python/).

Consider how code can be split in reusable building blocks, and write test cases for each unit in the `tests` folder. These tests can be re-run automatically, ensuring that the fundamental building blocks remain sound and can be improved and reused without hesitation. When you find yourself copy-pasting code from one place to another, this is likely a moment to stop and think if a new, reusable, function or class should be defined. Conversely, when setting out to change a commonly used function or class, make sure that adequate unit tests are in place.

#### Logging
While `print(var)` may be useful while testing new code, such statements should later be replaced by e.g.
```python
from examples import log

var = -1
log.debug(f'Just checking that this variable is {var}.')
log.info(f'The variable is {var}.')
log.warning(f'Be aware that variable is {var}!')
log.error(f'Hang on, the variable is {var}! This is not right!')
log.critical(f'Oh dear, the variable is {var}! I am quiting!')
```
This time-stamps the messages and records them to the file `optics.log` in the root folder. All messages are saved to
file. Messages with log-level `logging.DEBUG` are generally not shown on screen unless otherwise specified using e.g.
```python
import logging
from examples import log

log.level = logging.DEBUG
```
Messages with log-level `logging.INFO` are generally not shown when importing `log` from te `optics` module, except when 
importing log from the submodule `optics.experimental`. When writing a new module (a file that can be imported), use
instead
```python
import logging

log = logging.getLogger(__name__)
```
This will automatically pick up the configuration depending on its location. E.g. it will only show errors and
warnings when part of the `optics` package. To test such module, just reimport `log` at the start of your tests:
```python
import logging

log = logging.getLogger(__name__)

log.info('This is not visible on screen!')
log.warning('Only warnings or errors!')


if __name__ == '__main__':
    import logging
    from optics.experimental import log
    log.level = logging.DEBUG
    
    log.debug('Doing some tests now...')
```
