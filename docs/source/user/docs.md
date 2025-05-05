# Adding Documentation

## Building the Documentation
1. Install the project including documentation dependencies using `pip install -e .[dev]` from the root of the project.
2. Install Pandoc on your system using `sudo apt-get install pandoc` or `brew install pandoc`, or use the documentation here: [Pandoc Installation](https://pandoc.org/installing.html).
3. Go to the `/docs` directory and run `make <arg>`, depending on your goal:
    - `make html` to build the documentation.
    - `make server` to build the documentation and start a local server on http://localhost:8000 to view the documentation and rebuild automatically.
    - `make clean` to remove the built documentation.

## Structure of the Documentation
Currently, the documentation has two main sections: The {doc}`index` which serves as a written tutorial for the project and the {doc}`../reference/index` which serves as a technical reference for the project.

### The User Guide
This is a written tutorial for the project. It should be written in a way that a new user can understand and use the project. Background on methods and concepts can and should be included.

### The API Reference
This is a technical reference for the project. The user can go and look up the specifics of a method or class, including specific examples. This is generated to a big extent from the docstrings in the code. The developer mostly just needs to decide on the structure of this part of the documentation, and write the docstrings in the code accordingly.

## Technical Details
### How to Autogenerate Documentation
It is possible to autogenerate documentation from the docstrings in the code. We use this feature to generate the API reference. The docstrings should be written in reStructuredText format, using the [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

#### How to Add a New Section
First, create a new `.md` file in the `/docs/source/reference` directory and add the new file to the `toctree` in `/docs/source/reference/index.md`.

At the start of the file you can define the current module, which will be used to find the functions and classes in the module. This is done with the following code (the `>` characters are added to avoid execution of the directives):
````
> ```{eval-rst}
> .. currentmodule:: seismostats
> ```
````
You can then add the references to the functions and classes using the relative import path to the function
or class name.
````
> ```{eval-rst}
> .. autosummary::
>     :toctree: api/
> 
>     Catalog
> ```
```` 
This creates a list of references to the functions and classes, as well as, inside the `api` directory, a `.rst` file with the documentation for the functions and classes.

Optimally you would structure this documentation logically, and provide a minimal description of the groups
of functions and classes.

If you would like to have the documentation for a function or class directly in the current file, you can first use the `currentmodule` as above, and then the `autofunction` directive.
````
> ```{eval-rst}
> .. autofunction:: _example
> ```
```` 


### How to Use Crossreferences

#### Document
How to reference a document in the documentation.

**Code**  
```
Using a custom name:              {doc}`Reference <../reference/index>`  
Using the title of the document:  {doc}`../reference/index`
```

**Output**  
Using a custom name: {doc}`Reference <../reference/index>`  
Using the top level title of the document: {doc}`../reference/index`  

#### Reference
How to reference a section in the documentation.

**Code**  
```
Using a custom name:            {ref}`Reference <reference/catalog:modify catalog>`  
Using the title of the section: {ref}`/reference/catalog.md#modify-catalog`  
```

**Output**  
Using a custom name: {ref}`Reference <reference/formats/catalog:modify catalog>`  
Using the title of the section: {ref}`/reference/formats/catalog.md#modify-catalog`  

#### Function / Class
How to reference a function or class in the documentation.

**Code**  
```
Using a custom name:        {func}`Bin Magnitudes <seismostats.Catalog.bin_magnitudes>`  
Using the function name:    {func}`~seismostats.Catalog.bin_magnitudes`  
Or with the whole path:     {func}`seismostats.Catalog.bin_magnitudes`
```

**Output**  
Using a custom name: {func}`Bin Magnitudes <seismostats.Catalog.bin_magnitudes>`  
Using the title of the function: {func}`~seismostats.Catalog.bin_magnitudes`  
Or with the whole path:  {func}`seismostats.Catalog.bin_magnitudes`

#### Use Crossreferences in a Docstring
How to use crossreferences in a docstring.

**Code**  
```python
def _example():
    """
    This function is purely used as an example for the documentation.

    See Also:
        :func:`seismostats.Catalog.bin_magnitudes`
    """
    pass
```

**Output**  
```{eval-rst}
.. module:: seismostats.utils.docs
    :noindex:
```
```{eval-rst}
.. autofunction:: _example
```

### Add Mathematical Equations
Mathematical expressions can be added to the documentation using LaTeX. 
The expressions can be added in text or in docstrings.


#### In Text

**Code**  
```
$
\frac{\partial u}{\partial t}=-u \frac{\partial u}{\partial x}- \\
v \frac{\partial u}{\partial y}-w \frac{\partial u}{\partial z}
$
```

**Output**  
$
\frac{\partial u}{\partial t}=-u \frac{\partial u}{\partial x}- \\
v \frac{\partial u}{\partial y}-w \frac{\partial u}{\partial z}
$

#### In Text as Equation Blocks

**Code** 
````
```{math}
:label: mymath
(a + b)^2 = a^2 + 2ab + b^2
```

```{math}
:label: mymath2
(a + b)^2  &=  (a + b)(a + b) \\
           &=  a^2 + 2ab + b^2
```
The equation {eq}`mymath` is a quadratic equation, so is equation {eq}`mymath2`.
````

**Output**  
```{math}
:label: mymath
(a + b)^2 = a^2 + 2ab + b^2
```

```{math}
:label: mymath2
(a + b)^2  &=  (a + b)(a + b) \\
           &=  a^2 + 2ab + b^2
```
The equation {eq}`mymath` is a quadratic equation, so is equation {eq}`mymath2`.


#### In Docstrings

**Code**  
```python
def _math():
    """
    This function is purely used as an example for the documentation.

    Formulas inside docstrings can be used like this 
    :math:`\\frac{\\partial u}{\\partial t} = 1`. Note that backslashes 
    need to be escaped by doubling them up.
    """
    pass
```

**Output**  
```{eval-rst}
.. module:: seismostats.utils.docs
    :noindex:
```
```{eval-rst}
.. autofunction:: _math
```