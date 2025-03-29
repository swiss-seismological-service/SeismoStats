import pandas as pd

from seismostats import Catalog

simple_catalog = Catalog.from_dict({
    'longitude': [42.35, 1.35, 2.35],
    'latitude': [3.34444, 5.135, 2.134],
    'magnitude': [1.0, 2.5, 3.9]
})
simple_catalog.delta_m = 0.1
print(simple_catalog.estimate_mc_ks())
