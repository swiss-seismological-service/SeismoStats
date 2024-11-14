# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4

#
# LICENSE
#
# Copyright (C) 2010-2023 GEM Foundation, G. Weatherill, M. Pagani,
# D. Monelli.
#
# The Hazard Modeller's Toolkit is free software: you can redistribute
# it and/or modify it under the terms of the GNU Affero General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>
#
# DISCLAIMER
#
# The software Hazard Modeller's Toolkit (openquake.hmtk) provided herein
# is released as a prototype implementation on behalf of
# scientists and engineers working within the GEM Foundation (Global
# Earthquake Model).
#
# It is distributed for the purpose of open collaboration and in the
# hope that it will be useful to the scientific, engineering, disaster
# risk and software design communities.
#
# The software is NOT distributed as part of GEM’s OpenQuake suite
# (https://www.globalquakemodel.org/tools-products) and must be considered as a
# separate entity. The software provided herein is designed and implemented
# by scientific staff. It is not developed to the design standards, nor
# subject to same level of critical review by professional software
# developers, as GEM’s OpenQuake software suite.
#
# Feedback and contribution to the software is welcome, and can be
# directed to the hazard scientific staff of the GEM Model Facility
# (hazard@globalquakemodel.org).
#
# The Hazard Modeller's Toolkit (openquake.hmtk) is therefore distributed
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# The GEM Foundation, and the authors of the software, assume no
# liability for use of the software.

"""
Module :mod:`openquake.hmtk.seismicity.declusterer.dec_gardner_knopoff`
defines the Gardner and Knopoff declustering algorithm
"""

import numpy as np
import pandas as pd

from seismostats.analysis.declustering.base import BaseCatalogueDecluster
from seismostats.analysis.declustering.distance_time_windows import (
    BaseDistanceTimeWindow
)
from seismostats.analysis.declustering.utils import decimal_year, haversine

from typing import TypedDict


class GKConfig(TypedDict):
    time_distance_window: BaseDistanceTimeWindow
    fs_time_prop: float
    time_cutoff: int


class GardnerKnopoffType1(BaseCatalogueDecluster):
    """
    This class implements the Gardner Knopoff algorithm as described in
    this paper:
    Gardner, J. K. and Knopoff, L. (1974). Is the sequence of aftershocks
    in Southern California, with aftershocks removed, poissonian?. Bull.
    Seism. Soc. Am., 64(5): 1363-1367.
    """

    def decluster(self, catalogue: pd.DataFrame,
                  config: GKConfig) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply the Gardner-Knopoff declustering algorithm to the catalogue.

        The catalogue must contain the following columns:
        - time, magnitude, longitude, latitude

        The configuration dictionary must contain the following
        - time_distance_window: BaseDistanceTimeWindow
        - fs_time_prop: float in the interval [0,1], expressing
        the size of the time window used for searching for foreshocks,
        as a fractional proportion of the size of the aftershock window.
        Optional:
        - time_cutoff: for the time distance window

        If there are multiple events with the same magnitude,
        the earliest event is considered as the mainshock first.

        Args:
            catalogue: the catalogue of earthquakes
            config: configuration parameters

        Returns:
            cluster_ids: array with cluster numbers for each event
                         Events not in a cluster are assigned 0
            shock_types: array with shock types for each event
                        (foreshock=-1, mainshock=0, aftershock=+1)
        """
        # TODO in the end we want to work on the df directly
        # instead of converting to numpy arrays
        # magnitude = catalogue["magnitude"].to_numpy(dtype=np.float64)
        # longitude = catalogue["longitude"].to_numpy(dtype=np.float64)
        # latitude = catalogue["latitude"].to_numpy(dtype=np.float64)

        if "time" not in catalogue.columns:
            raise ValueError("Catalogue must contain a time column")

        year_dec = decimal_year(catalogue)
        neq = len(catalogue)  # Number of earthquakes

        # Pre-allocate cluster index vectors
        cluster_ids = np.zeros(neq, dtype=int)
        # Sort magnitudes into descending order
        original_magnitude = catalogue["magnitude"].to_numpy(dtype=np.float64)

        # todo: time instead
        # sorted_cat = catalogue.sort_values(by=["magnitude", "year", "month", "day"],
        #                                    ascending=False,
        #                                    kind="heapsort")
        # id0 = sorted_cat.index
        # sorted_cat.reindex()

        # id0 = pd.Series(original_magnitude).sort_values(ascending=False).index
        id0 = np.argsort(original_magnitude, kind="heapsort")[::-1]

        catalogue["__temp_time"] = year_dec
        id0 = catalogue.sort_values(by=["magnitude", "__temp_time"],
                                    ascending=False,
                                    kind="mergesort").index

        # Get space and time windows corresponding to each event
        # Initial Position Identifier
        sw_space, sw_time = config["time_distance_window"].calc(
            sorted_cat["magnitude"].values, config.get("time_cutoff")
        )
        # No need to reindex sw_space and sw_time since we pass the sorted magnitude
        # sw_space = sw_space[id0]
        # sw_time = sw_time[id0]
        # Of course we could also calculate decimal year after sorting
        year_dec = year_dec[id0]
        shock_types = np.zeros(neq, dtype=int)
        # Begin cluster identification
        clust_index = 0
        for i in range(0, neq - 1):  # TODO why neq - 1 instead of neq? (maybe the )
            # if cluster_ids[i] != 0:
            #     pass
            if cluster_ids[i] == 0:
                # Find Events inside both fore- and aftershock time windows
                dt = year_dec - year_dec[i]
                vsel = np.logical_and(
                    cluster_ids == 0,
                    np.logical_and(
                        dt >= (-sw_time[i] * config["fs_time_prop"]),
                        dt <= sw_time[i],
                    ),
                )
                # Of those events inside time window,
                # find those inside distance window
                vsel1 = (
                    haversine(
                        longitude[vsel],
                        latitude[vsel],
                        longitude[i],
                        latitude[i],
                    )
                    <= sw_space[i]
                )
                # wtf? (basically an and, updates the selection flag only where it was true, to the haversine result)
                vsel[vsel] = vsel1[:, 0]  # array of array

                # should be removable later
                temp_vsel = np.copy(vsel)
                temp_vsel[i] = False
                # A isolated mainshock does gets 0 as cluster number
                # TODO give a new ids (adapt tests)
                if any(temp_vsel):
                    # Allocate a cluster number
                    cluster_ids[vsel] = clust_index + 1
                    shock_types[vsel] = 1
                    # For those events in the cluster before the main event,
                    # flagvector is equal to -1
                    temp_vsel[dt >= 0.0] = False
                    shock_types[temp_vsel] = -1
                    shock_types[i] = 0
                    clust_index += 1

        # Re-sort the catalog_matrix into original order
        # id1 = np.argsort(eqid, kind="heapsort")

        # id0 is a permutation vector that sorts the catalogue
        # we can get the original order by inverting the permuation
        # This is O(n) instead of O(n log n) for sorting
        inverted = np.zeros(neq, dtype=int)
        inverted[id0] = np.arange(0, neq, 1)

        cluster_ids = cluster_ids[inverted]
        shock_types = shock_types[inverted]

        return cluster_ids, shock_types
