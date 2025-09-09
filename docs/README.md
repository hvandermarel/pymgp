# pyMGP - Documentation

Collection of Python modules for GNSS Positioning and Geodesy.

## Github pages

A description of the `pyMGP` modules can be found on [Gitlab pages](https://hvandermarel.github.io/pymgp/index.html).


## Updating documentation (locally)

This section is not for users.

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

## Updating documentation (Github pages)

Github actions is used to update the documentation on Github with every push. 
