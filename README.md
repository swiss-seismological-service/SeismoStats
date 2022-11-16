# catalog-tools

### Start developing:

```
# just use a basic virtual environment
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

### Use this repository inside another environment/code

```
# use it locally, being able to switch back and forth doing changes
# enter this in your other environment
pip install -e ../path/to/catalog-tools

# if you don't need to do live changes in catalog-tools you can install it "from source"
pip install git+ssh://git@github.com/swiss-seismological-service/catalog-tools.git

# if you want to install a specific branch:
pip install git+ssh://git@github.com/swiss-seismological-service/catalog-tools.git@feature/branch

# update it once the repo has changed:
pip install --force-reinstall git+ssh://git@github.com/swiss-seismological-service/catalog-tools.git
```
