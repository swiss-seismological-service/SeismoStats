# catalogue-tools

### Start developing:

```
# just use a basic virtual environment
python3 -m venv env

# activate it
source env/bin/activate 

# update build tools
pip install -U pip wheel setuptools

# install this package + requirements + development requirements ([dev])
pip install -e .[dev] 

# run tests
tox
```

### Use this repository inside another repository
```
# use it locally, being able to switch back and forth doing changes
# enter this in your other repository
pip install -e ../path/to/catalogue-tools

# if you don't need to do live changes in catalogue-tools you can install it "from source"
pip install git+ssh://git@github.com/swiss-seismological-service/catalogue-tools.git

# if you want to install a specific branch:
pip install git+ssh://git@github.com/swiss-seismological-service/catalogue-tools.git@feature/branch

# update it once the repo has changed:
pip install --force-reinstall git+ssh://git@github.com/swiss-seismological-service/catalogue-tools.git
```
