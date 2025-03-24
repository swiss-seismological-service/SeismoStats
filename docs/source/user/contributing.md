# Contributing

We welcome contributions to the project. Please read the following guidelines before submitting a pull request.

>[!IMPORTANT]  
>While we are still working on the first stable version, the contribution guidelines will still continue to change. We will update this documentation, as more and more parts of the project are finalized and ready for outside contributions.

## General
In General, please refer to existing implementations and adhere to the structure and style of the existing code.

## B-Value Estimation Methods
1. Create a new file inside the `seismostats.analysis.bvalue` package, containing a class which inherits from `seismostats.analysis.bvalue.BValueEstimator`.

```python	
class MyBValueEstimator(BValueEstimator):
```

2. Implement a constructor, which takes as a minimum `*args` and `**kwargs` and passes them on to `super().__init__(*args, **kwargs)`. This is necessary for the class to be able to be instantiated.

```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
```

If your method takes additional parameters, add them in the constructor and assign them to class variables

```python
def __init__(self, my_parameter: float, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.my_parameter = my_parameter
```

3. Implement the `_estimate` method, which does the actual calculation. This method does not take any input parameters, and returns the **b-value**. Please note the following points:
    * All necessary parameters are available as class variables. Just inheriting from the base class, will give you access to the following variables:
        * `self.magnitudes`: The magnitudes of the earthquakes.
        * `self.delta_m`: The magnitude bin width.
        * `self.mc`: The magnitude of completeness.
        * (optional) `self.weights`: The weights assigned to the magnitudes.
    * If you are filtering or modifying the `self.magnitudes` array, please make sure to update `self.magnitudes` with the final version of the magnitudes you are using.

```python	
def _estimate(self):
    # Do the calculation here
    return b_value
```

4. If your implementation allows using weights, set the `_weights_supported` class variable to `True`, or `False` otherwise.

```python
class MyBValueEstimator(BValueEstimator):
    _weights_supported = True
```

5. The basic functionalities should now work automatically, like calculation of the standard deviation or estimation of `beta` instead of `b`. You can still add additional methods to the class if you would like to, or submit a pull request to extend the base functionality of `BValueEstimator` if needed.

6. Add a simple regression test to the `tests` folder, which tests the functionality of your method. This is necessary to ensure that the method works correctly and to prevent future changes from breaking the method.  
The test should be simple, and only test the correct calculation of the b-value using a static dataset and b-value. If you added more functionality, you should also test this.


6.  Add docstrings, following the [Google Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings), this is necessary for the documentation to be generated correctly.

