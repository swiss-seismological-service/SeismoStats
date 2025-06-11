# Catalog


```{eval-rst}
.. currentmodule:: seismostats
```

## Catalog
Simple representation of an earthquake catalog, storing each single event as a row.

(constructor-cat)=
### Constructor

```{eval-rst}
.. autosummary::
    :toctree: ../api/
    :template: ../_templates/autosummary/class_notoctree.rst
    :nosignatures:

    Catalog
```
```{eval-rst}
.. autosummary::
    :toctree: ../api/
    :nosignatures:

    Catalog.from_quakeml
    Catalog.from_dict
```

### Estimate from Catalog

```{eval-rst}
.. autosummary::
    :toctree: ../api/
    :nosignatures:

    Catalog.estimate_b
    Catalog.estimate_a
    Catalog.estimate_mc_ks
    Catalog.estimate_mc_b_stability
    Catalog.estimate_mc_maxc
```

### Estimate from Catalog

```{eval-rst}
.. autosummary::
    :toctree: ../api/
    :nosignatures:

    Catalog.plot_in_space
    Catalog.plot_mags_in_time
    Catalog.plot_cum_count
    Catalog.plot_fmd
    Catalog.plot_cum_fmd
    Catalog.plot_mc_vs_b
```


### Modify Catalog

```{eval-rst}
.. autosummary::
    :toctree: ../api/
    :nosignatures:

    Catalog.bin_magnitudes
    Catalog.strip
    Catalog.drop_ids
    Catalog.drop_uncertainties
```


(convert-cat)=
### Convert from and to other format

```{eval-rst}
.. autosummary::
    :toctree: ../api/
    :nosignatures:

    Catalog.to_quakeml
    Catalog.from_quakeml
    Catalog.from_openquake
    Catalog.to_openquake
```