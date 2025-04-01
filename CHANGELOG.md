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

## [Unreleased]
<!-- Add your changes to the relevant section below, while uncommenting the section header. -->

#### Added
- API reference for a and b value packages, including full list of attributes.
- Thorough API documentation for the `Catalog` class.
- Added `estimate_mc_maxc` and `estimate_mc_bvalue_stability` methods to the `Catalog` class.

#### Changed
- `analysis.ks_test_gr` now takes the `b-value` as a parameter instead of `beta`.
- `Catalog.estimate_mc` got updated to the latest version of the `mc_ks` function and renamed to `Catalog.estimate_mc_ks`.
- Improve and streamline the `estimate_mc` functions and slightly change their input parameters.
- `Catalog.estimate_b` got updated to the class-based API of the `b_value` package.
- `Catalog.estimate_a` added to the Catalog class, including full documentation.

#### Fixed
- Build process for the documentation.
- API Reference Documentation

<!-- #### Removed -->

<!-- #### Deprecated -->


## [1.0.0rc0] - 2025-03-18
#### Added
- Initial release candidate of the library.
- Implemented core functionality.
- Provided first API version.
- Included documentation setup with ReadTheDocs.
