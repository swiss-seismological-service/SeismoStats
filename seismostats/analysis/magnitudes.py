import pandas as pd


def apply_edwards(mag_type: str, mag: float) -> pd.Series:
    """
    Converts local magnitudes to moment magnitudes according
    Edwards et al. (2010).

    Args:
        mag_type:   Magnitude type.
        mag:        Magnitude value.

    Returns:
        pd.Series:  Magnitude type and value.

    Source:
        Edwards, B., Allmann, B., Fäh, D., & Clinton, J. (2010).
        Automatic computation of moment magnitudes for small earthquakes
        and the scaling of local to moment magnitude.
        Geophysical Journal International, 183(1), 407-420.

    Examples:
        >>> from seismostats.analysis.magnitudes import apply_edwards
        >>> apply_edwards('ML', 3.0)
        0    Mw_converted
        1          2.8779

        >>> import pandas as pd
        >>> from seismostats import Catalog
        >>> data = {'longitude': [0, 1, 2],
        ...         'latitude': [0, 1, 2],
        ...         'depth': [0, 1, 2],
        ...         'time': pd.to_datetime(['2021-01-01 00:00:00',
        ...                                 '2021-01-01 10:00:00',
        ...                                 '2021-01-01 20:00:00']),
        ...         'magnitude': [1, 2, 3],
                    'mag_type': ['ML', 'Mw', 'Ml']}
        >>> catalog = Catalog(data)
        >>> for index, row in catalog.iterrows():
        ...     converted = apply_edwards(row['mag_type'], row['magnitude'])
        ...     catalog.at[index, 'mag_type'] = converted[0]
        ...     catalog.at[index, 'magnitude'] = converted[1]
        >>> catalog
        longitude	latitude	depth	time	magnitude	mag_type
        0	0	0	0	2021-01-01 00:00:00	1.5411	Mw_converted
        1	1	1	1	2021-01-01 10:00:00	2.0000	Mw
        2	2	2	2	2021-01-01 20:00:00	2.8779	Mw_converted

    """
    if "l" in mag_type.lower():
        return pd.Series(["Mw_converted", 1.02 + 0.472 * mag + 0.0491 * mag**2])
    elif "w" in mag_type.lower():
        return pd.Series([mag_type, mag])

    # usage
    # if convert_to_mw:
    #     cat[['mag_type', 'magnitude']] = cat.apply(
    #         lambda x: apply_edwards(x['mag_type'], x['magnitude']), axis=1)
