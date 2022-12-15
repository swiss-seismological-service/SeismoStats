"""All functions that are to be used externally are initialized here"""
# flake8: noqa

# analysis
from catalog_tools.analysis.estimate_beta import estimate_beta_elst
from catalog_tools.analysis.estimate_beta import estimate_beta_utsu
from catalog_tools.analysis.estimate_beta import estimate_beta_tinti

# download
from catalog_tools.download.download_catalogs import download_catalog_sed
from catalog_tools.download.download_catalogs import prepare_sed_catalog

# plots
from catalog_tools.plots.basics import plot_cum_fmd
from catalog_tools.plots.basics import plot_fmd

# utils
from catalog_tools.utils.binning import bin_magnitudes
from catalog_tools.utils.simulate_distributions import simulate_magnitudes
