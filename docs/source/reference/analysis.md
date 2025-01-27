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
```

## Estimating b-Values

```{eval-rst}
.. autosummary::
    :toctree: api/

    estimate_b
    analysis.bvalue.shi_bolt_confidence
    analysis.bvalue.ClassicBValueEstimator
    analysis.bvalue.BPositiveBValueEstimator
    analysis.bvalue.BMorePositiveBValueEstimator
    analysis.bvalue.UtsuBValueEstimator
```

## Estimating a-Values

```{eval-rst}
.. autosummary::
    :toctree: api/

    estimate_a
    analysis.estimate_a_classic
    analysis.estimate_a_positive
```

## Other
```{eval-rst}
.. autosummary::
    :toctree: api/

    analysis.bvalue.utils.make_more_incomplete
    analysis.bvalue.beta_to_b_value
    analysis.bvalue.b_value_to_beta
```