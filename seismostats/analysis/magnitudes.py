import pandas as pd


def apply_edwards(mag_type: str, mag: float) -> pd.Series:
    """
    Converts local magnitudes to moment magnitudes according to

    Edwards, B., Allmann, B., Fäh, D., & Clinton, J. (2010).
    Automatic computation of moment magnitudes for small earthquakes
    and the scaling of local to moment magnitude.
    Geophysical Journal International, 183(1), 407-420.
    """
    if "l" in mag_type.lower():
        return pd.Series(
            ['Mw_converted', 1.02 + 0.472 * mag + 0.0491 * mag ** 2])
    elif "w" in mag_type.lower():
        return pd.Series([mag_type, mag])

    # usage
    # if convert_to_mw:
    #     cat[['mag_type', 'magnitude']] = cat.apply(
    #         lambda x: apply_edwards(x['mag_type'], x['magnitude']), axis=1)
