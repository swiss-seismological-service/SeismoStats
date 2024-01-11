# Usage

## Installation

(installation)=

```bash
pip install -e .
python3 -m venv env

# activate it
source env/bin/activate

# update build tools
pip install -U pip wheel setuptools

# install this package + requirements + development requirements ([dev])
pip install -e '.[dev]'

# run tests
tox
```

## Code
You can use the {py:func}`seismostats.Catalog` class to load a catalog.