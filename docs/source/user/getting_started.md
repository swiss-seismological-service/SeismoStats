# Installation

`Seismostats` is a Python library but doesn't require extensive knowledge of Python. The statistical analysis of a catalog can be achieved easily without any coding expertise by following the step by step guide ({doc}`examples`).

## Required packages and libraries

We didn't reinvent the wheel and rely on existing libraries and packages to perform basic routines.

### GEOS
The plotting of the seismicity requires [GEOS](https://libgeos.org/), a C/C++ library for computational geometry. If `GEOS` is not installed on your machine, you will need to get it, for example on a linux machine with 
```console
sudo apt-get libgeos-dev
```
or on a mac with
```console
brew install geos
```

## Using SeismoStats in another code

### Install from source
This way of installing `SeismoStats` in another environement allows you to use the static version. 
```console
pip install git+https://github.com/swiss-seismological-service/SeismoStats.git
```

If you want to install a specific branch:
```console
pip install git+https://github.com/swiss-seismological-service/SeismoStats.git@feature/branch
```

To update your environment to the latest version of `SeismoStats`:
```console
pip install --force-reinstall git+https://github.com/swiss-seismological-service/SeismoStats.git
```