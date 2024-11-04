from copy import deepcopy

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import os
from seismostats import Catalog as SeismoCatalog
from seismostats.catalogs.catalog import REQUIRED_COLS_CATALOG
from seismostats.utils import _check_required_cols

pytest.importorskip("openquake.hmtk.seismicity.catalogue",
                    reason="Testing OpenQuake conversion requires\
                    the optional dependency openquake to be installed")
from openquake.hmtk.seismicity.catalogue import Catalogue as OQCatalog

PATH_RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'data')

COMMON_COLS = ['longitude', 'latitude', 'depth', 'magnitude']

# Note: when constructing a hmtk Catalogue from a dictionary,
# they keys must be already of the format expected by hmtk Catalogue
# hmtk Catalogue does not validate this and update its Data.
# Errors will appear only later


simple_df = SeismoCatalog({'longitude': [0, 1, 2],
                           'latitude': [0, 1, 2],
                           'depth': [0.12343, 13.535, 2.0],
                           'time': pd.to_datetime(['1950-01-01 00:00:00',
                                                   '1999-01-01 00:00:00',
                                                   '2023-01-01 00:00:00']),
                           'magnitude': [1, 2, 3]})


csv_file = os.path.join(PATH_RESOURCES, 'fdsnws2020.csv')
fdsnws = SeismoCatalog(pd.read_csv(csv_file))
# Currently the time column is not converted to datetime automatically
fdsnws["time"] = pd.to_datetime(fdsnws["time"])


simple_oq_catalogue = OQCatalog.make_from_dict({
    'eventID': ["event0", "event1", "event2"],
    'longitude': np.array([42.35, 1.35, 2.35], dtype=float),
    'latitude': np.array([3.34444, 5.135, 2.134], dtype=float),
    'depth': np.array([5.5, 10.52, 50.4], dtype=float),
    'year': np.array([1900, 1982, 2020], dtype=int),
    'month': np.array([1, 4, 11], dtype=int),
    'day': np.array([1, 7, 30], dtype=int),
    'hour': np.array([5, 7, 12], dtype=int),
    'minute': np.array([5, 7, 30], dtype=int),
    'second': np.array([13.1234, 15.0, 59.9999], dtype=float),
    'magnitude': np.array([1.0, 2.5, 3.9], dtype=float)
})

historical_oq_catalogue = OQCatalog.make_from_dict({
    'eventID': ["event0", "event1", "event2"],
    'longitude': np.array([42.35, 1.35, 2.35], dtype=float),
    'latitude': np.array([3.34444, 5.135, 2.134], dtype=float),
    'depth': np.array([5.5, 10.52, 50.4], dtype=float),
    'year': np.array([1200, 1333, 1600], dtype=int),
    'month': np.array([1, 4, 11], dtype=int),
    'day': np.array([1, 7, 30], dtype=int),
    'hour': np.array([5, 7, 12], dtype=int),
    'minute': np.array([5, 7, 30], dtype=int),
    'second': np.array([0, 15, 59], dtype=float),
    'magnitude': np.array([1.0, 2.5, 3.9], dtype=float)
})


@pytest.mark.parametrize("df", [simple_df, fdsnws])
def test_seismo_full_round(df: SeismoCatalog):
    converted = df.to_openquake()
    reconstructed = SeismoCatalog.from_openquake(converted)
    if 'eventID' not in df:
        reconstructed = reconstructed.drop(columns=['eventID'])
    print(df.compare(reconstructed))
    pdt.assert_frame_equal(df, reconstructed, rtol=1e-5, atol=1e-8)


HMTK_FLOAT_ATTRIBUTE_LIST = [
    "second",
    "timeError",
    "longitude",
    "latitude",
    "SemiMajor90",
    "SemiMinor90",
    "ErrorStrike",
    "depth",
    "depthError",
    "magnitude",
    "sigmaMagnitude",
]

HMTK_INT_ATTRIBUTE_LIST = ["year", "month", "day", "hour", "minute", "flag"]

HMTK_STRING_ATTRIBUTE_LIST = ["eventID", "Agency", "magnitudeType", "comment"]

HMTK_TOTAL_ATTRIBUTE_LIST = list(
    (set(HMTK_FLOAT_ATTRIBUTE_LIST).union(set(HMTK_INT_ATTRIBUTE_LIST))).union(
        set(HMTK_STRING_ATTRIBUTE_LIST)
    )
)


def compare_hmtk(cat1: OQCatalog, cat2: OQCatalog):
    data1 = cat1.data
    data2 = cat2.data
    for attribute in HMTK_TOTAL_ATTRIBUTE_LIST:
        has1 = attribute in data1 and len(data1[attribute]) > 0
        has2 = attribute in data2 and len(data2[attribute]) > 0
        print(attribute, has1, has2)
        assert has1 == has2 and "Attribute missing in one of the catalogs"
        if not has1 or not has2:
            continue
        col1 = data1[attribute]
        col2 = data2[attribute]
        if attribute in HMTK_FLOAT_ATTRIBUTE_LIST:
            np.testing.assert_allclose(col1, col2, rtol=1e-5, atol=1e-8)
        elif attribute in HMTK_INT_ATTRIBUTE_LIST:
            np.testing.assert_array_equal(col1, col2)
        else:
            assert col1 == col2 and "String attribute mismatch"


@pytest.mark.parametrize("oq_catalogue", [simple_oq_catalogue,
                                          historical_oq_catalogue])
def test_hmtk_full_round(oq_catalogue: OQCatalog):
    df = SeismoCatalog.from_openquake(oq_catalogue)
    reconstructed = df.to_openquake()
    compare_hmtk(oq_catalogue, reconstructed)


def test_to_openquake_simple():
    converted = simple_df.to_openquake()
    np.testing.assert_array_equal(converted['year'],
                                  np.array([1950, 1999, 2023]))
    np.testing.assert_array_equal(converted['month'],
                                  np.array([1, 1, 1]))
    np.testing.assert_array_equal(converted['day'],
                                  np.array([1, 1, 1]))
    for col in COMMON_COLS:
        np.testing.assert_array_almost_equal(converted[col],
                                             simple_df[col].to_numpy())


def test_from_openquake_simple():
    df = SeismoCatalog.from_openquake(simple_oq_catalogue)
    assert _check_required_cols(df, REQUIRED_COLS_CATALOG)
    for col in COMMON_COLS:
        np.testing.assert_allclose(df[col], simple_oq_catalogue[col],
                                   rtol=1e-5, atol=1e-8)


def test_from_openquake_modify():
    original_long = simple_oq_catalogue['longitude'].copy()
    df = SeismoCatalog.from_openquake(simple_oq_catalogue)
    df['longitude'] = 0
    assert (df['longitude'].to_numpy() == 0).all()
    assert (simple_oq_catalogue['longitude'] == original_long).all()


def test_to_openquake_modify():
    original_long = simple_df['longitude'].copy()
    converted = simple_df.to_openquake()
    converted.data['longitude'] = np.zeros(len(simple_df))
    assert (converted['longitude'] == 0).all()
    assert simple_df['longitude'].equals(original_long)


def test_to_openquake_missing_col():
    # 'longitude' missing
    df = simple_df.drop(columns=['longitude'])
    with pytest.raises(Exception):
        df.to_openquake()


def test_to_openquake_extra_col():
    extra_cols = ['extra1', 'extra2']
    df = simple_df.copy().assign(extra1=[1, 2, 3],
                                 extra2=["Hello", "World", "!"])
    converted = df.to_openquake()
    for col in extra_cols:
        assert df[col].equals(pd.Series(converted[col]))


def test_from_openquake_extra_col():
    data = deepcopy(simple_oq_catalogue.data)
    agencies = ["SED", "NA", "MarsQuakeService"]
    catalogue = OQCatalog.make_from_dict({**data, 'Agency': agencies})
    df = SeismoCatalog.from_openquake(catalogue)
    assert _check_required_cols(df, REQUIRED_COLS_CATALOG)
    assert (df['Agency'] == agencies).all()


def test_to_empty():
    df = SeismoCatalog()
    with pytest.raises(Exception):
        cat = df.to_openquake()
        assert cat.get_number_events() == 0


def test_from_empty():
    with pytest.raises(Exception):
        df = SeismoCatalog.from_openquake({})
        assert len(df) == 0
