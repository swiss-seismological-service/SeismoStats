"""All functions that are to be used externally are initialized here"""

# flake8: noqa

# analysis
from seismostats.analysis.estimate_beta import estimate_b, shi_bolt_confidence

# plots
from seismostats.plots.basics import (
    plot_cum_count,
    plot_cum_fmd,
    plot_fmd,
    plot_mags_in_time,
)
from seismostats.plots.seismicity import plot_in_space
from seismostats.plots.statistical import plot_mc_vs_b

# seismicity
from seismostats.seismicity.catalog import Catalog, ForecastCatalog
from seismostats.seismicity.rategrid import ForecastGRRateGrid, GRRateGrid
from seismostats.utils._config import get_option, set_option

# utils
from seismostats.utils.binning import bin_to_precision
from seismostats.utils.filtering import cat_intersect_polygon
from seismostats.utils.simulate_distributions import simulate_magnitudes
