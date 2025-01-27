# Copyright (C) 2010-2023 GEM Foundation, G. Weatherill, M. Pagani,
# D. Monelli.
# Hazard Modeller's Toolkit (openquake.hmtk) https://www.globalquakemodel.org

import unittest
import os
import numpy as np
import pandas as pd

from seismostats.analysis.declustering import (
    GardnerKnopoffType1,
    GardnerKnopoffWindow
)


class GardnerKnopoffType1TestCase(unittest.TestCase):
    """
    Unit tests for the Gardner and Knopoff declustering algorithm class.
    """

    BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

    def setUp(self):
        """
        Read the sample catalogue
        """
        flnme = "gardner_knopoff_test_catalogue.csv"
        filename = os.path.join(self.BASE_DATA_PATH, flnme)
        self.cat = pd.read_csv(filename)
        self.expected = self.cat["mainshock_flag"].to_numpy()

    def test_dec_gardner_knopoff(self):
        # Testing the Gardner and Knopoff algorithm
        tdw = GardnerKnopoffWindow()
        dec = GardnerKnopoffType1(time_distance_window=tdw,
                                  fs_time_prop=1.0)
        mainshock_flags = dec(self.cat)
        np.testing.assert_allclose(mainshock_flags, self.expected)

    def test_dec_gardner_knopoff_time_cutoff(self):
        """
        Testing the Gardner and Knopoff algorithm using a cutoff
        time of 100 days
        """
        tdw = GardnerKnopoffWindow(time_cutoff=100)
        dec = GardnerKnopoffType1(time_distance_window=tdw,
                                  fs_time_prop=1.0)
        mainshock_flags = dec(self.cat)
        expected = self.expected.copy()
        # event becomes mainshock when time_cutoff = 100
        expected[4] = True
        np.testing.assert_allclose(mainshock_flags, expected)
