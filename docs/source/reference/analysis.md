# Analysis

```{eval-rst}
.. currentmodule:: seismostats
```

## Estimating the Completeness Magnitude
```{eval-rst}
.. autosummary::
    :toctree: api/

    analysis.mc_ks
    analysis.mc_max_curvature
    analysis.mc_by_bvalue_stability
```

## Estimating b-Values

```{eval-rst}
.. autosummary::
    :toctree: api/

    analysis.estimate_b
    analysis.bvalue.shi_bolt_confidence
    analysis.bvalue.ClassicBValueEstimator
    analysis.bvalue.ClassicBValueEstimator.calculate
    analysis.bvalue.BPositiveBValueEstimator
    analysis.bvalue.BMorePositiveBValueEstimator
    analysis.bvalue.UtsuBValueEstimator
```

## Estimating a-Values

```{eval-rst}
.. autosummary::
    :toctree: api/

    analysis.estimate_a
    analysis.ClassicAValueEstimator
    analysis.APositiveAValueEstimator
    analysis.AMorePositiveAValueEstimator
```

## B-Significant
```{eval-rst}
.. autosummary::
    :toctree: api/

    analysis.b_significant_1D
```

## Other
```{eval-rst}
.. autosummary::
    :toctree: api/

    analysis.bvalue.utils.make_more_incomplete
    analysis.bvalue.beta_to_b_value
    analysis.bvalue.b_value_to_beta
```