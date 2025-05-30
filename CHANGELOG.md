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
