from catalog_tools.rategrid import (REQUIRED_COLS_RATEGRID, ForecastRateGrid,
                                    RateGrid)

RAW_DATA = {
    'longitude_min': [-180, -90, 0, 90],
    'longitude_max': [-90, 0, 90, 180],
    'latitude_min': [-90, -45, 0, 45],
    'latitude_max': [-45, 0, 45, 90],
    'depth_min': [0, 10, 20, 30],
    'depth_max': [10, 20, 30, 40],
    'number_events': [100, 200, 300, 400],
    'a': [1.0, 1.5, 2.0, 2.5],
    'b': [0.5, 0.6, 0.7, 0.8],
    'mc': [4.0, 4.5, 5.0, 5.5],
    'grid_id': [1, 2, 3, 4]
}


def test_rategrid_init():
    # Test initialization with data
    data = {'name': ['Object 1', 'Object 2', 'Object 3'],
            'a': [10.0, 12.5, 8.2]}
    rategrid = RateGrid(data)
    assert isinstance(rategrid, RateGrid)
    assert rategrid.name is None

    # Test initialization with name
    rategrid = RateGrid(data, name='My RateGrid')
    assert rategrid.name == 'My RateGrid'

    # Test initialization with additional arguments
    rategrid = RateGrid(data, columns=['name', 'a'])


def test_forecast_rategrid_init():
    # Test initialization with data
    data = {'name': ['Object 1', 'Object 2', 'Object 3'],
            'a': [10.0, 12.5, 8.2]}
    rategrid = ForecastRateGrid(data)
    assert isinstance(rategrid, ForecastRateGrid)


def test_rategrid_strip():
    # Test stripping columns
    rategrid = RateGrid(RAW_DATA)
    stripped_rategrid = rategrid.strip()
    assert isinstance(stripped_rategrid, RateGrid)
    assert stripped_rategrid.columns.tolist().sort() == \
        REQUIRED_COLS_RATEGRID.sort()

    # Test inplace stripping
    rategrid.strip(inplace=True)
    assert rategrid.columns.tolist().sort() == REQUIRED_COLS_RATEGRID.sort()

    # Test constructor fallback
    dropped = rategrid.drop(columns=['a'])
    assert not isinstance(dropped, RateGrid)


def test_forecast_rategrid_strip():
    # Test stripping columns
    rategrid = ForecastRateGrid(RAW_DATA)
    stripped_rategrid = rategrid.strip()
    assert isinstance(stripped_rategrid, ForecastRateGrid)

    # Test constructor fallback "downgrade"
    dropped = rategrid.drop(columns=['grid_id'])
    assert isinstance(dropped, RateGrid)
