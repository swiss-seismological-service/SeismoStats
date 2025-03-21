# Copyright (C) 2010-2023 GEM Foundation, G. Weatherill, M. Pagani,
# D. Monelli.
# Hazard Modeller's Toolkit (openquake.hmtk) https://www.globalquakemodel.org

import numpy as np
import pandas as pd

from seismostats.analysis.declustering.base import Declusterer
from seismostats.analysis.declustering.distance_time_windows import \
    BaseDistanceTimeWindow
from seismostats.analysis.declustering.utils import haversine


class GardnerKnopoffType1(Declusterer):
    """
    Implements the Gardner Knopoff declustering algorithm.

    Source:
        Gardner, J. K. and Knopoff, L. (1974). Is the sequence of aftershocks
        in Southern California, with aftershocks removed, poissonian?. Bull.
        Seism. Soc. Am., 64(5): 1363-1367.

    Args:
        time_distance_window:
                            :class:`seismostats.analysis.BaseDistanceTimeWindow`
        fs_time_prop:           Float in the interval [0,1], expressing
                            the size of the time window used for searching
                            for foreshocks, as a fractional proportion of
                            the size of the aftershock window.
    """

    def __init__(
        self,
        time_distance_window: BaseDistanceTimeWindow,
        fs_time_prop: float = 1.0,
    ):
        super().__init__()
        self.time_distance_window = time_distance_window
        self.fs_time_prop = fs_time_prop

    def _decluster(
        self,
        catalog: pd.DataFrame,
    ) -> np.ndarray[np.bool_]:
        """
        Applies the Gardner-Knopoff declustering algorithm to the catalog.

        The catalog must contain the following columns:
        - time, magnitude, longitude, latitude

        If there are multiple events with the same magnitude,
        the earliest event is considered as the mainshock first.

        Args:
            catalog: The catalog of earthquakes.

        Returns:
            mainshock_flags: Boolean array indicating whether the i'th event
                             is a mainshock.

        Raises:
            ValueError: Indicates that a required column is missing.
        """
        cols = set(("time", "magnitude", "longitude", "latitude"))
        if not cols.issubset(set(catalog.columns)):
            raise ValueError(
                "catalog must contain the following columns: " + ", ".join(cols)
            )

        # each cluster of events is assigned a non-negative integer id
        cluster_ids = np.zeros(len(catalog), dtype=int)
        cluster_id = 1
        magnitude = catalog["magnitude"]
        longitude = catalog["longitude"]
        latitude = catalog["latitude"]

        time = catalog["time"].astype("datetime64[s]", errors="ignore").values
        space_windows, time_windows = self.time_distance_window(magnitude)
        ordered = catalog[list(cols)].sort_values(
            by=["magnitude", "time"], ascending=[False, True], kind="mergesort"
        )
        mainshock_flags = np.ones(len(catalog), dtype=bool)
        for i, long, lat in zip(
            ordered.index, ordered["longitude"], ordered["latitude"]
        ):
            # If already assigned to a cluster, skip
            if cluster_ids[i] != 0:
                continue

            # Find Events inside both fore- and aftershock time windows
            dt = time - time[i]
            vsel = np.logical_and(
                cluster_ids == 0,
                np.logical_and(
                    dt >= (-time_windows[i] * self.fs_time_prop),
                    dt <= time_windows[i],
                ),
            )
            # Of those events inside time window,
            # find those inside the distance window
            vsel1 = (
                haversine(
                    longitude[vsel],
                    latitude[vsel],
                    long,
                    lat,
                )
                <= space_windows[i]
            )
            vsel[vsel] = vsel1
            # Assign id and flags to this cluster
            cluster_ids[vsel] = cluster_id
            cluster_id += 1
            mainshock_flags[vsel] = 0
            mainshock_flags[i] = 1

        self.__cluster_ids = cluster_ids
        return mainshock_flags
