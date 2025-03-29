from importlib.metadata import version

__version__ = version("seismostats")
from seismostats.catalogs.catalog import Catalog, ForecastCatalog  # noqa
from seismostats.catalogs.client import FDSNWSEventClient  # noqa
from seismostats.catalogs.rategrid import ForecastGRRateGrid  # noqa
from seismostats.catalogs.rategrid import GRRateGrid  # noqa
