from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Declusterer(ABC):
    """
    Abstract base class for the implementation of declustering algorithms.
    """

    def __int__(self):
        self.__cluster_ids = None
        self.__results = None

    @abstractmethod
    def _decluster(self, catalog: pd.DataFrame) -> np.ndarray[np.bool_]:
        """
        Implements the declustering algorithm.

        Args:
            catalog: Earthquake catalog to be declustered.
        """
        return NotImplemented

    def __call__(self, catalog: pd.DataFrame) -> np.ndarray[np.bool_]:
        """
        Declusters the catalog.

        Args:
            catalog: Earthquake catalog to be declustered.

        Returns:
            mainshock_flags:    Boolean array indicating whether the i'th event
                            is a mainshock.
        """
        self.__results = self._decluster(catalog)
        return self.__results
