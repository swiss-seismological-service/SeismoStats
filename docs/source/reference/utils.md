# Utils

```{eval-rst}
.. currentmodule:: seismostats
```

## Binning

```{eval-rst}
.. autosummary::
    :toctree: api/

    bin_to_precision
    utils.get_fmd
    utils.get_cum_fmd

```

## Synthetic Magnitude Distributions

```{eval-rst}
.. autosummary::
    :toctree: api/

    utils.simulate_magnitudes
    utils.simulate_magnitudes_binned

```

## Coordinates

### Spatial Transformations
```{eval-rst}
.. autosummary::
    :toctree: api/

    utils.bounding_box_to_polygon
    utils.polygon_to_bounding_box

```
### Coordinate Transformer
```{eval-rst}
.. autosummary::
    :toctree: api/
    
    utils.CoordinateTransformer
    utils.CoordinateTransformer.to_local_coords
    utils.CoordinateTransformer.from_local_coords
    utils.CoordinateTransformer.polygon_from_local_coords
    utils.CoordinateTransformer.polygon_to_local_coords
```

## Spatial Filtering

```{eval-rst}
.. autosummary::
    :toctree: api/

    utils.cat_intersect_polygon

```