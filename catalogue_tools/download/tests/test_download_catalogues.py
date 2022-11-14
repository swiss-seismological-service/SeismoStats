import datetime as dt
from numpy.testing import assert_equal
import numpy as np

from catalogue_tools.download.download_catalogues import download_catalog_sed


def test_download_catalogue_sed():
    min_mag = 3.0
    start_time = dt.datetime(1900, 1, 1)
    end_time = dt.datetime(2022, 1, 1)

    # download the CH catalog
    ch_cat = download_catalog_sed(
        start_time=start_time,
        end_time=end_time,
        min_magnitude=min_mag,
        only_earthquakes=False
    )

    # check that the downloaded catalog is correct
    assert_equal([
        len(ch_cat), len(ch_cat.query("event_type != 'earthquake'"))
    ], [1274, 18])

    assert_equal(
        np.unique(ch_cat["MagType"], return_counts=True),
        (['MLh', 'MLhc', 'Ml', 'Mw'],
         [93, 15, 32, 1134])
    )


if __name__ == '__main__':
    test_download_catalogue_sed()
