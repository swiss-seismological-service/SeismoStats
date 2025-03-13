# SeismoStats

Measure your seismicity with **SeismoStats**, a Python package for seismicity analysis.

>[!IMPORTANT]  
>We are actively working on a first stable version of `SeismoStats`. The API is not final yet and will still change until the first release coming in the next months. We are happy to receive feedback and suggestions for improvement.

## Start using `SeismoStats`:

This is intended for people interested in using existing functionalities and functions in `SeismoStats`, for example if you want to calculate an Mc, a-value and b-value for your catalogue and plot the frequency magnitude distribution.

```
# On your machine, select and go to the folder you want to have SeismoStats installed:
cd Folder_seismostats/

# Install the latest main version of SeismoStats "from source"
git clone https://github.com/swiss-seismological-service/SeismoStats.git .

# Install core dependencies
pip install .

# That's all, you can now use SeismoStats!
```

Note: this section is subject to changes soon, as we are working on getting `SeismoStats` into the [Python Package Index](https://pypi.org/).

## Start developing:

This is intended for people interested in contributing to the code base, for example by implementing a different method for the calculation of a magnitude of completeness or b-value. If you only want to use existing functions, please see the previous section.
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

## Use this repository inside another environment/code

```
# use it locally, being able to switch back and forth doing changes
# enter this in your other environment
pip install -e ../path/to/SeismoStats

# if you don't need to do live changes in catalog-tools you can install it "from source"
pip install git+https://github.com/swiss-seismological-service/SeismoStats.git

# if you want to install a specific branch:
pip install git+https://github.com/swiss-seismological-service/SeismoStats.git@feature/branch

# update it once the repo has changed:
pip install --force-reinstall git+https://github.com/swiss-seismological-service/SeismoStats.git
```

## Citing
We are actively working on a publication to submit with the first stable version of `SeismoStats`. If you use the code for scientific work, and until a pre-print is available, please cite `SeismoStats` as:

Mirwald, A., Schmid, N., Han, M., Rohnacher, A., Mizrahi, L., Ritz, V. A., & Wiemer, S. (2025). SeismoStats: A Python Package for Statistical Seismology. https://github.com/swiss-seismological-service/SeismoStats

```
@misc{Mirwald2025,
   author = {Aron Mirwald and Nicolas Schmid and Marta Han and Alicia Rohnacher and Leila Mizrahi and Vanille A. Ritz and Stefan Wiemer},
   title = {SeismoStats: A Python Package for Statistical Seismology},
   url = {https://github.com/swiss-seismological-service/SeismoStats},
   year = {2025}
}
```
