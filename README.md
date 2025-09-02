# pyMGP 

Collection of Python modules for GNSS Positioning and Geodesy.

## Description

A description of the modules can be found on [Gitlab pages](https://hvandermarel.github.io/pymgp/index.html).


## Installation

Activate the virtual environment to install into.

To install directly from Github

```
pip install git+https://github.com/hvandermarel/pymgp.git
```

To install from (a cloned) repository, change directory to the root folder of the repository with the `myprojects.toml` file and use pip to install

```
pip install .
```

To install an editable development version (using links to the repository) use

```
pip install -e .
```

To uninstall 

```
pip uninstall pymgp
```

## Updating documentation

Provisional, nor for users, move to its own document...

### One time initialization

A few additional packages are needed for the sphinx theme and Jupyter notebook support
``` 
conda install pydata-sphinx-theme
conda install nbsphinx --channel conda-forge
``` 

One time initialization of the `docs` directory

```
cd docs
sphinx-quickstart
```
The files `conf.py` and `index.rst` in `docs/source` have been modified. 

Newly added files are `api.doc`, `usage.rst`, `release.rst` and `examples.rst` (all referenced from `index.rst`). 

### API documentation updates

The file `api.rst` contains links to `reference/<module>.rst` which contains the `autosummary` tags.

The function documentation (in `docs/reference/generated` is updated by the commands

```
sphinx-autogen .\docs\source\reference\crstrans.rst
sphinx-autogen .\docs\source\reference\satorb.rst
sphinx-autogen .\docs\source\reference\tleorb.rst
sphinx-autogen .\docs\source\reference\orbplot.rst
```

### Build html documentation

The html documentation is made with

```
make html
```
or
```
make clean html
make html
```


