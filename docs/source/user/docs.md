# Creating and Building the Documentation

## Building the Documentation
1. Install the project including documentation dependencies using `pip install -e .[doc]` from the root of the project.
2. Go to the `/docs` directory and run `make <arg>`, depending on your goal:
    - `make html` to build the documentation.
    - `make server` to build the documentation and start a local server on http://localhost:8000 to view the documentation and rebuild automatically.
    - `make clean` to remove the built documentation.

## Writing the Documentation
Currently, the documentation has two main sections: The {doc}`index` which serves as a written tutorial for the project and the {doc}`../reference/index` which serves as a technical reference for the project.

### The User Guide
This is a written tutorial for the project. It should be written in a way that a new user can understand and use the project. Background on methods and concepts can and should be included.

### The API Reference
This is a technical reference for the project. The user can go and look up the specifics of a method or class, including specific examples. This is generated to a big extent from the docstrings in the code. The developer mostly just needs to decide on the structure of this part of the documentation, and write the docstrings in the code accordingly.