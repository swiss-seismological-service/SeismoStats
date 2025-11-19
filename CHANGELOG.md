# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- Template for a new unreleased block, copy, do not uncomment -------------------------- -->

<!-- ## [Unreleased] -->
<!-- Add your changes to the relevant section below, while uncommenting the section header. -->

<!-- #### Added -->

<!-- #### Changed -->

<!-- #### Fixed -->

<!-- #### Removed -->

<!-- #### Deprecated -->
<!-- -------------------------------------------------------------------------------------- -->

## [v1.0.0] - 2025-11-20

#### Added
- b-significant test for evaluating spatial and temporal significance of b-value variations, including plotting functionality and comprehensive documentation.
- Lilliefors test implementation for goodness-of-fit testing of the Gutenberg-Richter distribution.
- Bootstrap method to `BValue` class for uncertainty estimation of b-value calculations.
- Weights support for Kolmogorov-Smirnov test (`ks_test_gr`) and frequency-magnitude distribution plotting functions to handle weighted catalogs.
- `value` property for both `AValue` and `BValue` estimator classes for consistent access to estimated values.
- `std` property for `AValue` estimator class to access standard deviation of a-value estimates.
- `is_estimated` property and `n` property to both `AValue` and `BValue` classes for better consistency between the two estimator classes.
- Safety mechanism for `values_from_partitioning` to handle edge cases more robustly.
- `get_options` function to module initialization for accessing configuration options.
- Comprehensive test coverage for a-value estimators, b-significant test, Lilliefors test, and Kolmogorov-Smirnov test.

#### Changed
- Same as `get_cum_fmd`, `plot_cum_fmd` (both function and `Catalog` method) now take `fmd_bin` as input instead of `delta_m`, to reflect that this argument doesn't necessarily match the discretization of magnitudes, and be consistent with `get_fmd` and `plot_fmd` which already have this modification.
- Optimized maximum magnitude finding in `estimate_mc` functions for improved performance.
- Optimized simulation of truncated exponential distribution for better computational efficiency.
- Restricted bootstrap method to only work within the b-more-positive estimation method where it is statistically applicable.
- Improved robustness of a-value estimation to handle edge cases better.
- Adjusted Lilliefors test (`p_lillierfors`) to be compatible with the b-positive estimation method.
- Bootstrap standard deviation calculation now orders magnitudes by time and removes NaN values for more accurate uncertainty estimates.
- Improved colors and labels in plotting functions for better visualization and consistency.
- Adjusted plotting functions to produce fewer warnings during normal usage.
- Reorganized example notebooks: removed redundant catalog notebook and cleaned up the how-to notebook for better clarity.
- User documentation and API reference extensively updated to reflect new features and improvements.

#### Fixed
- Binning precision issues by introducing epsilon tolerance to avoid incorrect magnitude binning due to floating-point arithmetic.
- Rounding behavior in binning functions that was inconsistent with numpy's `round()` behavior, ensuring consistent bin assignment.
- Error message for edge case when `delta_m < epsilon` to provide clearer feedback.
- Bug in `moransI` spatial analysis function.
- Axis input handling for seismicity plotting functions to work correctly with custom axes.
- Minor error in bootstrap calculation that could affect uncertainty estimates.
- Various typos in documentation and docstrings.


## [1.0.0rc2] - 2025-05-30
Add your changes to the relevant section below, while uncommenting the section header.
#### Added
- API and User Documentation has been extensively updated and improved.
- Added methods to calculate statistics on the `RateGrid` class and concatenate multiple `RateGrid` objects.

#### Changed
- Small changes to the `plots` module api to make it more consistent.
- Updated the example jupyter notebooks to reflect the new API.
- `estimate_mc` and `plot_fmd` take now `fmd_bin` as input instead of `delta_m`, to reflect that this is independent from the discretization of the catalog magnitudes.
- Functions and methods to estimate the `mc` value now return a consistent format. Instead of a tuple with a variable amount of values, they now return a tuple with first the `best_mc` value and then a dictionary with the additional values.

#### Fixed
- In the `binning_test`, shift the array to the smallest value, to avoid that the bin-center has an effect on the test.

## [1.0.0rc1] - 2025-04-09
#### Added
- API reference for a and b value packages, including full list of attributes.
- Thorough API documentation for the `Catalog` class.
- Added `estimate_mc_maxc` and `estimate_mc_b_stability` methods to the `Catalog` class.
- Option to download catalogs in batches from FDSNWS in order to avoid timeouts.

#### Changed
- `analysis.ks_test_gr` now takes the `b-value` as a parameter instead of `beta`.
- `Catalog.estimate_mc` got updated to the latest version of the `mc_ks` function and renamed to `Catalog.estimate_mc_ks`.
- Improve and streamline the `estimate_mc` functions and slightly change their input parameters.
- `Catalog.estimate_b` got updated to the class-based API of the `b_value` package.
- `Catalog.estimate_a` added to the Catalog class, including full documentation.

#### Fixed
- Build process for the documentation.
- API Reference Documentation

## [1.0.0rc0] - 2025-03-18
#### Added
- Initial release candidate of the library.
- Implemented core functionality.
- Provided first API version.
- Included documentation setup with ReadTheDocs.
